from typing import List, Dict
from src.schedulers.base import BaseScheduler
from src.state.band_state import BandState

class APSScheduler(BaseScheduler):
    """Adaptive Predictive Spectrum Scheduler."""
    
    def __init__(self, weights: Dict[str, float] = None):
        # Default weights for the scoring function
        self.weights = weights or {
            'w_D': 1.0, # Coverage Debt
            'w_A': 1.0, # Activity
            'w_P': 1.0, # Persistence
            'w_N': 1.0, # Novelty
            'w_U': 1.0, # Uncertainty
            'w_M': 1.0, # Miss Risk
            'w_C': 1.0, # Scan Cost
        }

    def compute_score(self, state: BandState) -> float:
        """Calculates the priority score for a single band."""
        score = (
            self.weights.get('w_D', 0) * state.coverage_debt +
            self.weights.get('w_A', 0) * state.activity +
            self.weights.get('w_P', 0) * state.persistence +
            self.weights.get('w_N', 0) * state.novelty +
            self.weights.get('w_U', 0) * state.uncertainty +
            self.weights.get('w_M', 0) * state.miss_risk -
            self.weights.get('w_C', 0) * state.scan_cost
        )
        return score

    def select_next(self, states: List[BandState]) -> int:
        if not states:
            return -1
        
        # Transparent scoring mechanism
        best_state = max(states, key=self.compute_score)
        return best_state.band_id
