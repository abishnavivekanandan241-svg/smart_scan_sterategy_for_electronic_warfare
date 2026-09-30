from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class BandState:
    """Maintains the APSS state for a specific frequency band."""
    band_id: int
    activity: float = 0.0
    persistence: float = 0.0
    novelty: float = 0.0
    uncertainty: float = 1.0 # High uncertainty initially
    coverage_debt: float = 0.0
    miss_risk: float = 0.0
    scan_cost: float = 1.0
    
    last_scan_time: float = 0.0
    last_result: bool = False
    pulse_count: int = 0
    hit_count: int = 0
    miss_count: int = 0
    
    # New PDW-derived statistics
    recent_activity: float = 0.0
    avg_amplitude: float = 0.0
    avg_pw: float = 0.0
    avg_aoa: float = 0.0

    def update_after_scan(self, current_time: float, hit: bool):
        """Update state immediately after a scan."""
        self.last_scan_time = current_time
        self.last_result = hit
        self.coverage_debt = 0.0 # Reset debt after scanning
        self.uncertainty *= 0.5 # Uncertainty drops after observing
        
        if hit:
            self.hit_count += 1
            self.activity = min(1.0, self.activity + 0.2)
            self.persistence = min(1.0, self.persistence + 0.1)
            self.miss_risk = 0.0
        else:
            self.miss_count += 1
            self.activity = max(0.0, self.activity - 0.2)
            self.persistence = max(0.0, self.persistence - 0.1)

    def update_from_pdw_stats(self, current_time: float, stats: Dict[str, float]):
        """
        Update state using aggregated PDW statistics from the preprocessor.
        This provides richer state updates than just a boolean hit/miss.
        """
        self.last_scan_time = current_time
        self.coverage_debt = 0.0
        self.uncertainty *= 0.5
        
        observed_pulse_count = stats.get("pulse_count", 0)
        observed_activity = stats.get("activity", 0.0)
        
        # Calculate novelty transparently: difference between historical activity and current observation
        self.novelty = abs(observed_activity - self.recent_activity)
        
        # Update persistence: Exponential Moving Average (EMA) of activity over time
        alpha = 0.3
        self.persistence = (alpha * observed_activity) + ((1.0 - alpha) * self.persistence)
        
        # Store for next novelty calculation
        self.recent_activity = observed_activity
        
        if observed_pulse_count > 0:
            self.last_result = True
            self.hit_count += 1
            self.pulse_count += int(observed_pulse_count)
            self.activity = observed_activity
            self.miss_risk = 0.0
            
            # Update physical stats
            self.avg_amplitude = stats.get("avg_amplitude", 0.0)
            self.avg_pw = stats.get("avg_pw", 0.0)
            self.avg_aoa = stats.get("avg_aoa", 0.0)
        else:
            self.last_result = False
            self.miss_count += 1
            self.activity = 0.0

    def update_time(self, current_time: float):
        """Update state due to passage of time (without scanning)."""
        time_since_scan = current_time - self.last_scan_time
        self.coverage_debt = min(1.0, self.coverage_debt + 0.05)
        self.uncertainty = min(1.0, self.uncertainty + 0.02)
        if self.activity > 0 or self.recent_activity > 0:
            eff_activity = max(self.activity, self.recent_activity)
            self.miss_risk = min(1.0, self.miss_risk + 0.05 * eff_activity)
