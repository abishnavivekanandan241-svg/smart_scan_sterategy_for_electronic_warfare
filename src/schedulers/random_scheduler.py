import random
from typing import List
from src.schedulers.base import BaseScheduler
from src.state.band_state import BandState

class RandomScheduler(BaseScheduler):
    """Selects the next band purely at random."""
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def select_next(self, states: List[BandState]) -> int:
        if not states:
            return -1
        return self.rng.choice(states).band_id
