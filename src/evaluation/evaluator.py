from typing import Dict, Any
from src.schedulers.base import BaseScheduler
from src.environment.rf_env import RFEnvironment
from src.state.band_state import BandState

class Evaluator:
    """Runs the simulation loop linking Environment, State, and Scheduler."""
    
    def __init__(self, env: RFEnvironment, scheduler: BaseScheduler, num_bands: int):
        self.env = env
        self.scheduler = scheduler
        self.num_bands = num_bands
        self.states = [BandState(band_id=i) for i in range(num_bands)]
        self.time = 0.0
        self.total_hits = 0
        self.total_scans = 0

    def run(self, steps: int, dwell_time: float = 1.0) -> Dict[str, Any]:
        """Execute the evaluation loop for a given number of steps."""
        for _ in range(steps):
            # 1. Update time for all states and environment
            self.time += dwell_time
            self.env.step(dwell_time)
            for state in self.states:
                state.update_time(self.time)
                
            # 2. Scheduler picks a band
            target_band = self.scheduler.select_next(self.states)
            if target_band == -1:
                continue
                
            # 3. Environment returns scan result
            hit = self.env.scan(target_band, dwell_time)
            self.total_scans += 1
            if hit:
                self.total_hits += 1
                
            # 4. Update the state of the scanned band
            self.states[target_band].update_after_scan(self.time, hit)
            
        return {
            "total_scans": self.total_scans,
            "total_hits": self.total_hits,
            "hit_rate": self.total_hits / max(1, self.total_scans)
        }
