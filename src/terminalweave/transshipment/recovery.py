from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Mapping, Sequence

from terminalweave.core.clock import ensure_utc
from terminalweave.core.errors import CapacityError, ValidationError
from terminalweave.transshipment.connections import Connection, ConnectionGraph, ServiceCall
from terminalweave.transshipment.dwell import DwellRecord
from terminalweave.transshipment.rebooking import RebookingPlanner, Sailing
from terminalweave.yard.inventory import InventoryLedger


class ContainerStage(str, Enum):
    IN_YARD = 'in_yard'
    WAITING = 'waiting'
    LOADED = 'loaded'
    DELIVERED = 'delivered'


ContainerStatus = ContainerStage
ContainerState = ContainerStage


class RecoveryStatus(str, Enum):
    RECOVERED = 'recovered'
    PARTIAL = 'partial'
    FAILED = 'failed'
    NOT_REQUIRED = 'not_required'


@dataclass(frozen=True)
class ScheduleChange:
    """A changed service call. ``revised_call=None`` cancels the call."""

    call_id: str
    revised_call: ServiceCall | None


@dataclass(frozen=True)
class RecoverySelection:
    container_id: str
    inbound_call: str
    original_outbound_call: str
    outbound_call: str
    sailing: Sailing
    teu: int
    new_connection: Connection

    @property
    def sailing_id(self) -> str:
        return self.sailing.sailing_id

    @property
    def new_outbound_call(self) -> str:
        return self.outbound_call


@dataclass(frozen=True)
class RecoveryFailure:
    container_id: str
    inbound_call: str
    outbound_call: str
    reason: str
    reason_code: str | None = None

    @property
    def code(self) -> str | None:
        return self.reason_code


@dataclass(frozen=True)
class ConnectionRecoveryPlan:
    recovered: tuple[RecoverySelection, ...]
    failures: tuple[RecoveryFailure, ...]
    unaffected: tuple[Connection, ...]
    affected: tuple[Connection, ...]

    @property
    def status(self) -> RecoveryStatus:
        if not self.affected:
            return RecoveryStatus.NOT_REQUIRED
        if self.failures and self.recovered:
            return RecoveryStatus.PARTIAL
        if self.failures:
            return RecoveryStatus.FAILED
        return RecoveryStatus.RECOVERED

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def successful(self) -> bool:
        return self.ok

    @property
    def partial(self) -> bool:
        return self.status is RecoveryStatus.PARTIAL

    @property
    def affected_containers(self) -> tuple[str, ...]:
        return tuple(connection.container_id for connection in self.affected)

    @property
    def recovered_container_ids(self) -> tuple[str, ...]:
        return tuple(selection.container_id for selection in self.recovered)

    @property
    def failed_container_ids(self) -> tuple[str, ...]:
        return tuple(failure.container_id for failure in self.failures)

    @property
    def recovered_count(self) -> int:
        return len(self.recovered)

    @property
    def failure_count(self) -> int:
        return len(self.failures)

    @property
    def new_connections(self) -> tuple[Connection, ...]:
        return tuple(selection.new_connection for selection in self.recovered)


_IMMUTABLE_STAGES = {
    ContainerStage.LOADED.value,
    ContainerStage.DELIVERED.value,
    'on_board',
    'shipped',
    'loaded_on_board',
    'vessel_loaded',
    'gate_out',
}

_ELIGIBLE_STAGES = {
    ContainerStage.IN_YARD.value,
    ContainerStage.WAITING.value,
    'yard',
    'discharged',
    'awaiting_load',
}


class ConnectionRecoveryPlanner:
    """Build rebooking actions for transshipment connections broken by a timetable."""

    def __init__(
        self,
        graph: ConnectionGraph,
        statuses: Mapping[str, str | ContainerStage] | None = None,
        ledger: InventoryLedger | None = None,
        dwell_records: Sequence[DwellRecord] | None = None,
        rebooker: RebookingPlanner | None = None,
    ):
        if not isinstance(graph, ConnectionGraph):
            raise ValidationError('connection graph required')
        self.graph = graph
        self.statuses = dict(statuses or {})
        self.ledger = ledger
        self.dwell_records: dict[str, DwellRecord] = {}
        for record in dwell_records or []:
            if record.container_id in self.dwell_records:
                raise ValidationError(f'duplicate dwell record for {record.container_id}')
            self.dwell_records[record.container_id] = record
        self.rebooker = rebooker or RebookingPlanner()

    def plan(
        self,
        changes: Sequence[ScheduleChange] | Mapping[str, ServiceCall | None] | ScheduleChange,
        sailings: Sequence[Sailing],
        *,
        as_of: datetime | None = None,
        max_wait: timedelta = timedelta(days=7),
        teu_by_container: Mapping[str, int] | None = None,
        destinations: Mapping[str, str] | None = None,
    ) -> ConnectionRecoveryPlan:
        revised_calls = self._normalize_changes(changes)
        if as_of is not None:
            as_of = ensure_utc(as_of)
        if max_wait.total_seconds() < 0:
            raise ValidationError('maximum wait cannot be negative')

        teu_by_container = dict(teu_by_container or {})
        for container_id, teu in teu_by_container.items():
            if not isinstance(teu, int) or teu <= 0:
                raise ValidationError(f'TEU for {container_id} must be a positive integer')

        sailings = list(sailings)
        if len({s.sailing_id for s in sailings}) != len(sailings):
            raise ValidationError('duplicate candidate sailing')
        destinations = dict(destinations or {})

        affected = []
        unaffected = []
        for connection in self.graph.connections:
            if self._connection_affected(connection, revised_calls):
                affected.append(connection)
            else:
                unaffected.append(connection)

        recovered: list[RecoverySelection] = []
        failures: list[RecoveryFailure] = []
        allocated_teu: dict[str, int] = {}

        for connection in affected:
            container_id = connection.container_id
            stage = self._stage(container_id)
            if stage in _IMMUTABLE_STAGES:
                failures.append(
                    self._failure(connection, f'container already {stage}', 'immutable_container')
                )
                continue
            if stage not in _ELIGIBLE_STAGES:
                failures.append(
                    self._failure(connection, f'unknown container stage: {stage}', 'unknown_stage')
                )
                continue

            revised_inbound = revised_calls.get(
                connection.inbound_call, self.graph.calls[connection.inbound_call]
            )
            revised_outbound = revised_calls.get(
                connection.outbound_call, self.graph.calls[connection.outbound_call]
            )
            if revised_inbound is None:
                failures.append(
                    self._failure(connection, 'inbound service call canceled', 'inbound_canceled')
                )
                continue

            destination = self._destination_for(connection, revised_outbound, destinations)
            if not destination:
                failures.append(
                    self._failure(connection, 'destination unknown', 'destination_unknown')
                )
                continue

            ready_at = (
                revised_inbound.arrival
                + connection.minimum_transfer
                + connection.yard_buffer
            )
            teu = teu_by_container.get(container_id, 1)
            preferred_service = self.graph.calls[connection.outbound_call].service
            available = [
                sailing
                for sailing in sailings
                if (sailing.call_id or sailing.sailing_id) != revised_inbound.call_id
                and self._candidate_current(sailing, connection, revised_calls)
                and self._candidate_time_allowed(sailing, ready_at, as_of, max_wait)
                and sailing.remaining_teu - allocated_teu.get(sailing.sailing_id, 0) >= teu
            ]
            try:
                chosen = self.rebooker.choose(
                    destination,
                    ready_at,
                    teu,
                    available,
                    preferred_service=preferred_service,
                    max_wait=max_wait,
                )
            except CapacityError:
                failures.append(
                    self._failure(
                        connection,
                        'no eligible recovery sailing',
                        'no_eligible_sailing',
                    )
                )
                continue

            allocated_teu[chosen.sailing_id] = allocated_teu.get(chosen.sailing_id, 0) + teu
            outbound_call_id = chosen.call_id or chosen.sailing_id
            new_connection = Connection(
                container_id,
                revised_inbound.call_id,
                outbound_call_id,
                connection.minimum_transfer,
                connection.yard_buffer,
            )
            recovered.append(
                RecoverySelection(
                    container_id,
                    revised_inbound.call_id,
                    connection.outbound_call,
                    outbound_call_id,
                    chosen,
                    teu,
                    new_connection,
                )
            )

        return ConnectionRecoveryPlan(
            recovered=tuple(recovered),
            failures=tuple(failures),
            unaffected=tuple(unaffected),
            affected=tuple(affected),
        )

    def recover(self, *args, **kwargs) -> ConnectionRecoveryPlan:
        return self.plan(*args, **kwargs)

    def affected_connections(
        self,
        changes: Sequence[ScheduleChange] | Mapping[str, ServiceCall | None] | ScheduleChange,
    ) -> tuple[Connection, ...]:
        revised_calls = self._normalize_changes(changes)
        return tuple(
            connection
            for connection in self.graph.connections
            if self._connection_affected(connection, revised_calls)
        )

    def affected_containers(
        self,
        changes: Sequence[ScheduleChange] | Mapping[str, ServiceCall | None] | ScheduleChange,
    ) -> tuple[str, ...]:
        return tuple(
            connection.container_id for connection in self.affected_connections(changes)
        )

    @staticmethod
    def _failure(connection: Connection, reason: str, reason_code: str) -> RecoveryFailure:
        return RecoveryFailure(
            connection.container_id,
            connection.inbound_call,
            connection.outbound_call,
            reason,
            reason_code,
        )

    @staticmethod
    def _normalize_changes(
        changes: Sequence[ScheduleChange] | Mapping[str, ServiceCall | None] | ScheduleChange,
    ) -> dict[str, ServiceCall | None]:
        if isinstance(changes, ScheduleChange):
            items = [(changes.call_id, changes.revised_call)]
        elif isinstance(changes, Mapping):
            items = list(changes.items())
        else:
            items = [(change.call_id, change.revised_call) for change in changes]

        revised: dict[str, ServiceCall | None] = {}
        for call_id, revised_call in items:
            if not call_id:
                raise ValidationError('service call id required')
            if call_id in revised:
                raise ValidationError(f'duplicate schedule change for {call_id}')
            if revised_call is not None and not isinstance(revised_call, ServiceCall):
                raise ValidationError('revised schedule must be a service call')
            revised[call_id] = revised_call
        return revised

    def _connection_affected(
        self,
        connection: Connection,
        revised_calls: Mapping[str, ServiceCall | None],
    ) -> bool:
        if (
            connection.inbound_call not in revised_calls
            and connection.outbound_call not in revised_calls
        ):
            return False

        revised_inbound = revised_calls.get(
            connection.inbound_call, self.graph.calls[connection.inbound_call]
        )
        revised_outbound = revised_calls.get(
            connection.outbound_call, self.graph.calls[connection.outbound_call]
        )
        if revised_inbound is None or revised_outbound is None:
            return True

        ready = revised_inbound.arrival + connection.minimum_transfer + connection.yard_buffer
        return ready > revised_outbound.departure

    def _stage(self, container_id: str) -> str:
        if self.ledger is not None:
            history = self.ledger.history(container_id)
            if history:
                last = history[-1]
                if last.event_type == 'loaded':
                    return ContainerStage.LOADED.value
                if last.event_type == 'gate_out':
                    return ContainerStage.DELIVERED.value
                return ContainerStage.IN_YARD.value

        dwell = self.dwell_records.get(container_id)
        if dwell is not None:
            return (
                ContainerStage.LOADED.value
                if dwell.loaded_at is not None
                else ContainerStage.IN_YARD.value
            )

        stage = self.statuses.get(container_id, ContainerStage.IN_YARD.value)
        return stage.value if isinstance(stage, Enum) else str(stage).lower()

    def _destination_for(
        self,
        connection: Connection,
        revised_outbound: ServiceCall | None,
        destinations: Mapping[str, str],
    ) -> str | None:
        destination = destinations.get(connection.container_id)
        if destination:
            return destination
        destination = destinations.get(connection.outbound_call)
        if destination:
            return destination
        if revised_outbound is not None and revised_outbound.destination:
            return revised_outbound.destination
        return self.graph.calls[connection.outbound_call].destination

    @staticmethod
    def _candidate_current(
        sailing: Sailing,
        connection: Connection,
        revised_calls: Mapping[str, ServiceCall | None],
    ) -> bool:
        candidate_call = sailing.call_id or sailing.sailing_id
        if candidate_call == connection.outbound_call:
            revised_outbound = revised_calls.get(connection.outbound_call)
            if revised_outbound is not None and revised_outbound.call_id != connection.outbound_call:
                return False

        if candidate_call not in revised_calls:
            return True
        revised_call = revised_calls[candidate_call]
        if revised_call is None:
            return False
        return (
            sailing.departure == revised_call.departure
            and sailing.service == revised_call.service
        )

    @staticmethod
    def _candidate_time_allowed(
        sailing: Sailing,
        ready_at: datetime,
        as_of: datetime | None,
        max_wait: timedelta,
    ) -> bool:
        if sailing.departure < ready_at or sailing.departure - ready_at > max_wait:
            return False
        return as_of is None or sailing.departure >= as_of
