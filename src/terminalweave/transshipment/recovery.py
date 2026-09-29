from __future__ import annotations
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import Enum
from terminalweave.core.clock import ensure_utc
from terminalweave.core.errors import ValidationError, ConflictError, CapacityError, NotFoundError
from .connections import Connection, ConnectionGraph, ServiceCall
from .rebooking import RebookingPlanner, Sailing


class ContainerStatus(Enum):
    """Operational state of a transshipment box at the terminal."""
    EXPECTED = 'expected'
    IN_YARD = 'in_yard'
    LOADED = 'loaded'
    DELIVERED = 'delivered'


# States that make a connection physically impossible to rework.
LOCKED_STATUSES = frozenset({ContainerStatus.LOADED, ContainerStatus.DELIVERED})


class InfeasibilityReason(Enum):
    INBOUND_LATE = 'inbound_late'
    OUTBOUND_EARLY = 'outbound_early'
    BOTH = 'both'
    # Each leg still meets the other side's original timing on its own, but the
    # combined movement swallowed the connection slack.
    SCHEDULE_COMPRESSION = 'schedule_compression'


class FailureReason(Enum):
    LOCKED_STATE = 'locked_state'
    MISSING_CONTEXT = 'missing_context'
    NO_ELIGIBLE_SAILING = 'no_eligible_sailing'


@dataclass(frozen=True)
class ScheduleUpdate:
    """A revised arrival/departure for a service call."""
    call_id: str
    arrival: datetime
    departure: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, 'arrival', ensure_utc(self.arrival))
        object.__setattr__(self, 'departure', ensure_utc(self.departure))
        if self.departure <= self.arrival:
            raise ValidationError('service call departure must follow arrival')


@dataclass(frozen=True)
class ShipmentContext:
    """What the terminal knows about a box that is transshipping."""
    container_id: str
    status: ContainerStatus
    destination: str
    teu: int

    def __post_init__(self) -> None:
        if self.teu <= 0:
            raise ValidationError('TEU must be positive')
        if not self.destination:
            raise ValidationError('destination is required')


@dataclass(frozen=True)
class AffectedConnection:
    container_id: str
    inbound_call: str
    outbound_call: str
    ready_at: datetime
    reason: InfeasibilityReason


@dataclass(frozen=True)
class RebookingDecision:
    container_id: str
    inbound_call: str
    sailing: Sailing
    ready_at: datetime
    teu: int
    wait: timedelta


@dataclass(frozen=True)
class RecoveryFailure:
    container_id: str
    reason: FailureReason
    detail: str


class RecoveryIncompleteError(ConflictError):
    """Raised when a recovery batch still has unresolved boxes."""

    def __init__(self, failures: list[str], locked: list[str]):
        self.failures = list(failures)
        self.locked = list(locked)
        parts = []
        if failures:
            parts.append('no recovery: ' + ', '.join(failures))
        if locked:
            parts.append('cannot rework: ' + ', '.join(locked))
        super().__init__('; '.join(parts) or 'recovery incomplete')


@dataclass
class RecoveryPlan:
    """Result of evaluating one schedule revision against the connection graph."""
    affected: list[AffectedConnection]
    rebooked: list[RebookingDecision]
    failures: list[RecoveryFailure]
    revised_calls: dict[str, ServiceCall]

    @property
    def affected_ids(self) -> list[str]:
        return [a.container_id for a in self.affected]

    @property
    def rebooked_ids(self) -> list[str]:
        return [r.container_id for r in self.rebooked]

    @property
    def failed_ids(self) -> list[str]:
        return [f.container_id for f in self.failures if f.reason is FailureReason.NO_ELIGIBLE_SAILING]

    @property
    def locked_ids(self) -> list[str]:
        return [f.container_id for f in self.failures if f.reason is FailureReason.LOCKED_STATE]

    @property
    def missing_ids(self) -> list[str]:
        return [f.container_id for f in self.failures if f.reason is FailureReason.MISSING_CONTEXT]

    @property
    def unresolved_ids(self) -> list[str]:
        return [f.container_id for f in self.failures]

    @property
    def complete(self) -> bool:
        return not self.failures

    def replacement_for(self, container_id: str, original: Connection) -> Connection | None:
        for decision in self.rebooked:
            if decision.container_id == container_id:
                return replace(original, outbound_call=decision.sailing.sailing_id)
        return None

    def summary(self) -> dict[str, object]:
        return {
            'affected': len(self.affected),
            'rebooked': len(self.rebooked),
            'failed': self.failed_ids,
            'locked': self.locked_ids,
            'missing_context': self.missing_ids,
            'complete': self.complete,
        }


class SailingLedger:
    """Tracks TEU already committed to candidate sailings during recovery.

    A dry-run plan never mutates this ledger; ``reserve`` is only called when a
    plan is committed, so planning stays deterministic and repeatable.
    """

    def __init__(self, sailings: list[Sailing]):
        ids = [s.sailing_id for s in sailings]
        if len(ids) != len(set(ids)):
            raise ValidationError('duplicate sailing id')
        self.sailings: dict[str, Sailing] = {s.sailing_id: s for s in sailings}

    def get(self, sailing_id: str) -> Sailing:
        if sailing_id not in self.sailings:
            raise NotFoundError(f'unknown sailing: {sailing_id}')
        return self.sailings[sailing_id]

    def remaining(self, sailing_id: str) -> int:
        return self.get(sailing_id).remaining_teu

    def reserve(self, sailing_id: str, teu: int) -> Sailing:
        if teu <= 0:
            raise ValidationError('TEU must be positive')
        sailing = self.get(sailing_id)
        if sailing.remaining_teu < teu:
            raise ConflictError(f'sailing {sailing_id} short of capacity')
        updated = replace(sailing, booked_teu=sailing.booked_teu + teu)
        self.sailings[sailing_id] = updated
        return updated

    def release(self, sailing_id: str, teu: int) -> Sailing:
        sailing = self.get(sailing_id)
        updated = replace(sailing, booked_teu=max(0, sailing.booked_teu - teu))
        self.sailings[sailing_id] = updated
        return updated


class RecoveryPlanner:
    """Builds connection recovery plans after schedule revisions."""

    def __init__(self, graph: ConnectionGraph, ledger: SailingLedger | None = None,
                 rebooker: RebookingPlanner | None = None):
        self.graph = graph
        self.ledger = ledger or SailingLedger([])
        self.rebooker = rebooker or RebookingPlanner()

    def revised_schedule(self, updates: list[ScheduleUpdate]) -> dict[str, ServiceCall]:
        if len({u.call_id for u in updates}) != len(updates):
            raise ValidationError('duplicate schedule update for call')
        revised = dict(self.graph.calls)
        for update in updates:
            if update.call_id not in revised:
                raise ValidationError(f'unknown service call: {update.call_id}')
            revised[update.call_id] = ServiceCall(
                update.call_id,
                revised[update.call_id].service,
                update.arrival,
                update.departure,
            )
        return revised

    def plan(self, updates: list[ScheduleUpdate], contexts: list[ShipmentContext],
             now: datetime, max_wait: timedelta = timedelta(days=7)) -> RecoveryPlan:
        now = ensure_utc(now)
        if max_wait.total_seconds() < 0:
            raise ValidationError('max wait cannot be negative')
        by_container = self._contexts(contexts)
        revised = self.revised_schedule(updates)

        affected: list[AffectedConnection] = []
        for connection in self.graph.connections:
            reason = self._infeasibility(connection, revised)
            if reason is None:
                continue
            inbound = revised[connection.inbound_call]
            ready = inbound.arrival + connection.minimum_transfer + connection.yard_buffer
            affected.append(AffectedConnection(
                connection.container_id,
                connection.inbound_call,
                connection.outbound_call,
                ready,
                reason,
            ))
        affected.sort(key=lambda a: (a.ready_at, a.container_id))

        # Tentative capacity view shared by every selection in this batch, so a
        # box that wins an earlier slot cannot double-book the same sailing.
        tentative: dict[str, int] = {}
        rebooked: list[RebookingDecision] = []
        failures: list[RecoveryFailure] = []

        for item in affected:
            context = by_container.get(item.container_id)
            if context is None:
                failures.append(RecoveryFailure(
                    item.container_id, FailureReason.MISSING_CONTEXT,
                    'no shipment context for affected container'))
                continue
            if context.status in LOCKED_STATUSES:
                failures.append(RecoveryFailure(
                    item.container_id, FailureReason.LOCKED_STATE,
                    f'container is {context.status.value} and cannot be reworked'))
                continue
            ready_at = max(now, item.ready_at)
            candidates = [
                self._with_hold(s, tentative)
                for s in self.ledger.sailings.values()
                if s.sailing_id != item.outbound_call
            ]
            try:
                chosen = self.rebooker.choose(
                    context.destination, ready_at, context.teu, candidates,
                    max_wait=max_wait)
            except CapacityError:
                failures.append(RecoveryFailure(
                    item.container_id, FailureReason.NO_ELIGIBLE_SAILING,
                    f'no sailing to {context.destination} within {max_wait}'))
                continue
            tentative[chosen.sailing_id] = tentative.get(chosen.sailing_id, 0) + context.teu
            rebooked.append(RebookingDecision(
                item.container_id, item.inbound_call,
                self.ledger.get(chosen.sailing_id), ready_at, context.teu,
                chosen.departure - ready_at))

        return RecoveryPlan(affected, rebooked, failures, revised)

    def commit(self, plan: RecoveryPlan) -> dict[str, str]:
        """Reserve capacity for every decision in a plan.

        Fails closed and rolls back any partial reservations if the ledger no
        longer has room, so a batch is never half-booked silently.
        """
        reserved: dict[str, int] = {}
        sailing_by_container: dict[str, str] = {}
        try:
            for decision in plan.rebooked:
                self.ledger.reserve(decision.sailing.sailing_id, decision.teu)
                reserved[decision.sailing.sailing_id] = (
                    reserved.get(decision.sailing.sailing_id, 0) + decision.teu)
                sailing_by_container[decision.container_id] = decision.sailing.sailing_id
        except ConflictError:
            for sailing_id, teu in reserved.items():
                self.ledger.release(sailing_id, teu)
            raise
        return sailing_by_container

    def ensure_complete(self, plan: RecoveryPlan) -> None:
        failed = [f.container_id for f in plan.failures
                  if f.reason is FailureReason.NO_ELIGIBLE_SAILING]
        locked = [f.container_id for f in plan.failures
                  if f.reason is FailureReason.LOCKED_STATE]
        missing = [f.container_id for f in plan.failures
                   if f.reason is FailureReason.MISSING_CONTEXT]
        if failed or locked or missing:
            raise RecoveryIncompleteError(failed + missing, locked)

    def _contexts(self, contexts: list[ShipmentContext]) -> dict[str, ShipmentContext]:
        by_container: dict[str, ShipmentContext] = {}
        for context in contexts:
            if context.container_id in by_container:
                raise ValidationError(f'duplicate shipment context: {context.container_id}')
            by_container[context.container_id] = context
        return by_container

    def _with_hold(self, sailing: Sailing, tentative: dict[str, int]) -> Sailing:
        held = tentative.get(sailing.sailing_id, 0)
        if held:
            return replace(sailing, booked_teu=sailing.booked_teu + held)
        return sailing

    def _infeasibility(self, connection: Connection,
                       revised: dict[str, ServiceCall]) -> InfeasibilityReason | None:
        inbound = revised[connection.inbound_call]
        outbound = revised[connection.outbound_call]
        transfer = connection.minimum_transfer + connection.yard_buffer
        original_inbound = self.graph.calls[connection.inbound_call]
        original_outbound = self.graph.calls[connection.outbound_call]

        currently_feasible = (
            original_inbound.arrival + transfer <= original_outbound.departure)
        if not currently_feasible:
            return None
        if inbound.arrival + transfer <= outbound.departure:
            return None
        inbound_late = inbound.arrival + transfer > original_outbound.departure
        outbound_early = original_inbound.arrival + transfer > outbound.departure
        if inbound_late and outbound_early:
            return InfeasibilityReason.BOTH
        if inbound_late:
            return InfeasibilityReason.INBOUND_LATE
        if outbound_early:
            return InfeasibilityReason.OUTBOUND_EARLY
        return InfeasibilityReason.SCHEDULE_COMPRESSION
