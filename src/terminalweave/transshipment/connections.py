from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from terminalweave.core.clock import ensure_utc
from terminalweave.core.errors import ValidationError, ConflictError

@dataclass(frozen=True)
class ServiceCall:
    call_id: str
    service: str
    arrival: datetime
    departure: datetime
    destination: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'arrival', ensure_utc(self.arrival))
        object.__setattr__(self, 'departure', ensure_utc(self.departure))
        if self.departure <= self.arrival:
            raise ValidationError('service call departure must follow arrival')

@dataclass(frozen=True)
class Connection:
    container_id: str
    inbound_call: str
    outbound_call: str
    minimum_transfer: timedelta
    yard_buffer: timedelta = timedelta()

    def __post_init__(self) -> None:
        if self.minimum_transfer.total_seconds() < 0 or self.yard_buffer.total_seconds() < 0:
            raise ValidationError('transfer durations cannot be negative')

class ConnectionGraph:

    def __init__(self):
        self.calls: dict[str, ServiceCall] = {}
        self.connections: list[Connection] = []

    def add_call(self, call: ServiceCall) -> None:
        if call.call_id in self.calls:
            raise ValidationError('duplicate service call')
        self.calls[call.call_id] = call

    def feasible(self, connection: Connection) -> bool:
        inbound = self.calls[connection.inbound_call]
        outbound = self.calls[connection.outbound_call]
        ready = inbound.arrival + connection.minimum_transfer + connection.yard_buffer
        return ready <= outbound.departure

    def add_connection(self, connection: Connection) -> None:
        if connection.inbound_call == connection.outbound_call:
            raise ConflictError('connection cannot loop to same call')
        if connection.inbound_call not in self.calls or connection.outbound_call not in self.calls:
            raise ValidationError('unknown service call')
        if any((c.container_id == connection.container_id for c in self.connections)):
            raise ConflictError('container already has a transshipment connection')
        if not self.feasible(connection):
            raise ConflictError('minimum connection time cannot be met')
        self.connections.append(connection)

    def service_edges(self) -> dict[str, set[str]]:
        edges = {call_id: set() for call_id in self.calls}
        for c in self.connections:
            edges[c.inbound_call].add(c.outbound_call)
        return edges

    def detect_cycle(self) -> list[str] | None:
        edges = self.service_edges()
        visiting = set()
        visited = set()
        stack = []

        def dfs(node: str):
            if node in visiting:
                idx = stack.index(node)
                return stack[idx:] + [node]
            if node in visited:
                return None
            visiting.add(node)
            stack.append(node)
            for nxt in sorted(edges[node]):
                cycle = dfs(nxt)
                if cycle:
                    return cycle
            stack.pop()
            visiting.remove(node)
            visited.add(node)
            return None
        for node in sorted(edges):
            cycle = dfs(node)
            if cycle:
                return cycle
        return None

    def validate_acyclic(self) -> None:
        cycle = self.detect_cycle()
        if cycle:
            raise ConflictError('service connection cycle: ' + ' -> '.join(cycle))

    def slack(self, connection: Connection) -> timedelta:
        inbound = self.calls[connection.inbound_call]
        outbound = self.calls[connection.outbound_call]
        ready = inbound.arrival + connection.minimum_transfer + connection.yard_buffer
        return outbound.departure - ready
