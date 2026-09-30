from typing import List, Dict, Any
from src.environment.pdw import PDW

class PDWPreprocessor:
    """Processes raw 5-D PDWs into band-level aggregated observations."""
    
    def __init__(self, num_bands: int, band_width: float, start_freq: float):
        self.num_bands = num_bands
        self.band_width = band_width
        self.start_freq = start_freq
        
    def get_band_id(self, cf: float) -> int:
        """Map Centre Frequency to a band ID transparently."""
        if cf < self.start_freq:
            return -1 # Out of bounds
        band = int((cf - self.start_freq) // self.band_width)
        if band >= self.num_bands:
            return -1
        return band

    def process_batch(self, pdws: List[PDW]) -> Dict[int, Dict[str, Any]]:
        """
        Processes a batch of PDWs into per-band aggregated observations.
        Returns a dict mapping band_id to a dictionary of features.
        """
        band_pdws = {i: [] for i in range(self.num_bands)}
        invalid_pw_counts = {i: 0 for i in range(self.num_bands)}
        out_of_bounds_count = 0
        
        for pdw in pdws:
            band_id = self.get_band_id(pdw.centre_frequency)
            if 0 <= band_id < self.num_bands:
                if pdw.pulse_width <= 0:
                    invalid_pw_counts[band_id] += 1
                else:
                    band_pdws[band_id].append(pdw)
            else:
                out_of_bounds_count += 1
                
        results = {}
        # Also return out of bounds count under a special key for tracking
        results[-1] = {"out_of_bounds_count": out_of_bounds_count}
        
        for band_id, b_pdws in band_pdws.items():
            pulse_count = len(b_pdws)
            inv_count = invalid_pw_counts[band_id]
            
            if pulse_count == 0:
                results[band_id] = {
                    "pulse_count": 0,
                    "activity": 0.0,
                    "avg_amplitude": 0.0,
                    "avg_pw": 0.0,
                    "avg_aoa": 0.0,
                    "invalid_pw_count": inv_count
                }
                continue
                
            # Basic physical aggregations (only on valid pdws)
            avg_amp = sum(p.amplitude for p in b_pdws) / pulse_count
            avg_pw = sum(p.pulse_width for p in b_pdws) / pulse_count
            avg_aoa = sum(p.aoa for p in b_pdws) / pulse_count
            
            # Transparent activity calculation
            # Normalize pulse_count up to a max expected threshold (e.g., 100)
            activity = min(1.0, pulse_count / 100.0)
            
            results[band_id] = {
                "pulse_count": pulse_count,
                "activity": activity,
                "avg_amplitude": avg_amp,
                "avg_pw": avg_pw,
                "avg_aoa": avg_aoa,
                "invalid_pw_count": inv_count
            }
            
        return results

    def batch_by_time_window(self, pdws: List[PDW], window_size: float):
        """
        Sorts PDWs chronologically by ToA and yields them in chunks of `window_size`.
        Yields: (window_start_time, window_end_time, pdws_in_window)
        """
        if not pdws:
            return
            
        # Ensure chronological order
        sorted_pdws = sorted(pdws, key=lambda p: p.toa)
        
        current_window_start = sorted_pdws[0].toa
        current_batch = []
        
        for pdw in sorted_pdws:
            if pdw.toa < current_window_start + window_size:
                current_batch.append(pdw)
            else:
                yield current_window_start, current_window_start + window_size, current_batch
                
                # Advance window start, skipping empty windows if gaps exist
                num_windows_skipped = int((pdw.toa - current_window_start) // window_size)
                current_window_start += num_windows_skipped * window_size
                
                current_batch = [pdw]
                
        if current_batch:
            yield current_window_start, current_window_start + window_size, current_batch
