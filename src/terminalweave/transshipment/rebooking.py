from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from terminalweave.core.clock import ensure_utc
from terminalweave.core.errors import ValidationError, CapacityError

@dataclass(frozen=True)
class Sailing:
    sailing_id: str
    service: str
    destination: str
    departure: datetime
    capacity_teu: int
    booked_teu: int = 0
    call_id: str | None = None

    def __post_init__(self):
        object.__setattr__(self, 'departure', ensure_utc(self.departure))
        if self.capacity_teu <= 0 or not 0 <= self.booked_teu <= self.capacity_teu:
            raise ValidationError('invalid sailing capacity')

    @property
    def remaining_teu(self) -> int:
        return self.capacity_teu - self.booked_teu

class RebookingPlanner:

    def choose(self, destination: str, ready_at: datetime, teu: int, sailings: list[Sailing], preferred_service: str | None=None, max_wait: timedelta=timedelta(days=7)) -> Sailing:
        ready_at = ensure_utc(ready_at)
        if teu <= 0:
            raise ValidationError('TEU must be positive')
        eligible = [s for s in sailings if s.destination == destination and s.departure >= ready_at and (s.departure - ready_at <= max_wait) and (s.remaining_teu >= teu)]
        if not eligible:
            raise CapacityError('no eligible rebooking sailing')
        return min(eligible, key=lambda s: (0 if preferred_service and s.service == preferred_service else 1, s.departure, -s.remaining_teu, s.sailing_id))
