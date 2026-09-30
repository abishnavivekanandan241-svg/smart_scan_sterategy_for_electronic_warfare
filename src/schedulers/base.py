from abc import ABC, abstractmethod
from typing import List
from src.state.band_state import BandState

class BaseScheduler(ABC):
    """Interface for all scan schedulers."""
    
    @abstractmethod
    def select_next(self, states: List[BandState]) -> int:
        """Selects the next band_id to scan."""
        pass
