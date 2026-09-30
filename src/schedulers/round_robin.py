from typing import List
from src.schedulers.base import BaseScheduler
from src.state.band_state import BandState

class RoundRobinScheduler(BaseScheduler):
    """Simple sequential scanner."""
    def __init__(self):
        self.current_idx = 0

    def select_next(self, states: List[BandState]) -> int:
        if not states:
            return -1
        band_id = states[self.current_idx].band_id
        self.current_idx = (self.current_idx + 1) % len(states)
        return band_id
