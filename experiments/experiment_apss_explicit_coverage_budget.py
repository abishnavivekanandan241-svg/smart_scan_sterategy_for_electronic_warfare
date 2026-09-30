import time
import math
from collections import defaultdict
from typing import Dict, Any

from src.preprocessing.tsrd_adapter import TSRDAdapter
from src.preprocessing.pdw_processor import PDWPreprocessor
from src.state.band_state import BandState
from src.schedulers.apss import APSScheduler

class ExperimentalBandState(BandState):
    debt_increment = 0.075
    uncertainty_increment = 0.02
    
    def update_time(self, current_time: float):
        self.coverage_debt = min(1.0, self.coverage_debt + self.__class__.debt_increment)
        self.uncertainty = min(1.0, self.uncertainty + self.__class__.uncertainty_increment)
        if self.activity > 0 or self.recent_activity > 0:
            eff_activity = max(self.activity, self.recent_activity)
            self.miss_risk = min(1.0, self.miss_risk + 0.05 * eff_activity)

class ExperimentalAPSScheduler(APSScheduler):
    def __init__(self, target_bands: int):
        super().__init__()
        self.target_bands = target_bands
        self.explored_bands = set()
        
    def select_next(self, states: list[BandState]) -> int:
        best_band = -1
        best_score = -float('inf')
        for i, state in enumerate(states):
            if len(self.explored_bands) >= self.target_bands and i not in self.explored_bands:
                continue
            score = self.compute_score(state)
            if score > best_score:
                best_score = score
                best_band = i
                
        if best_band != -1:
            self.explored_bands.add(best_band)
        return best_band

def get_score_components(state: BandState, weights: Dict[str, float]) -> Dict[str, float]:
    return {
        'D': weights.get('w_D', 1.0) * state.coverage_debt,
        'A': weights.get('w_A', 1.0) * state.activity,
        'P': weights.get('w_P', 1.0) * state.persistence,
        'N': weights.get('w_N', 1.0) * state.novelty,
        'U': weights.get('w_U', 1.0) * state.uncertainty,
        'M': weights.get('w_M', 1.0) * state.miss_risk,
        'C': -(weights.get('w_C', 1.0) * state.scan_cost)
    }

def evaluate_apss(name: str, target_coverage: float, pdws, num_bands: int, band_width: float, start_freq: float, window_size: float):
    processor = PDWPreprocessor(num_bands=num_bands, band_width=band_width, start_freq=start_freq)
    
    states = [ExperimentalBandState(band_id=i) for i in range(num_bands)]
    
    target_bands = math.ceil(target_coverage * num_bands)
    scheduler = ExperimentalAPSScheduler(target_bands=target_bands)
    weights = {'w_D': 1.0, 'w_A': 1.0, 'w_P': 1.0, 'w_N': 1.0, 'w_U': 1.0, 'w_M': 1.0, 'w_C': 1.0}
    scheduler.weights = weights
    
    total_scans = 0
    total_hits = 0
    scans_per_band = defaultdict(int)
    total_decision_time_ns = 0
    
    pending_pulses = defaultdict(list)
    latency_records = []
    dom_counts = defaultdict(int)
    
    for start_t, end_t, batch_pdws in processor.batch_by_time_window(pdws, window_size=window_size):
        for s in states:
            s.update_time(end_t)
            
        for p in batch_pdws:
            b_id = processor.get_band_id(p.centre_frequency)
            if 0 <= b_id < num_bands:
                pending_pulses[b_id].append(p.toa)
        
        t0 = time.perf_counter_ns()
        selected_band = scheduler.select_next(states)
        t1 = time.perf_counter_ns()
        total_decision_time_ns += (t1 - t0)
        
        if selected_band == -1:
            continue
            
        selected_state = states[selected_band]
        comps = get_score_components(selected_state, weights)
        dominant_comp = max(comps.items(), key=lambda x: x[1])[0]
        dom_counts[dominant_comp] += 1
            
        band_stats = processor.process_batch(batch_pdws)
        stats = band_stats.get(selected_band, {"pulse_count": 0, "activity": 0.0})
        is_hit = stats.get("pulse_count", 0) > 0
        
        if pending_pulses[selected_band]:
            for p_toa in pending_pulses[selected_band]:
                latency_records.append(end_t - p_toa)
            pending_pulses[selected_band].clear()
            
        states[selected_band].update_from_pdw_stats(end_t, stats)
        
        total_scans += 1
        scans_per_band[selected_band] += 1
        if is_hit:
            total_hits += 1

    unique_bands_scanned = len([b for b, c in scans_per_band.items() if c > 0])
    coverage = (unique_bands_scanned / num_bands) * 100 if num_bands > 0 else 0
    detection_rate = (total_hits / total_scans) * 100 if total_scans > 0 else 0
    avg_decision_time_ms = (total_decision_time_ns / total_scans) / 1_000_000 if total_scans > 0 else 0
    avg_latency = sum(latency_records) / len(latency_records) if latency_records else 0.0
    
    repeated_scans = total_scans - unique_bands_scanned
            
    return {
        "name": name,
        "total_scans": total_scans,
        "total_hits": total_hits,
        "detection_rate": detection_rate,
        "unique_bands_scanned": unique_bands_scanned,
        "repeated_scans": repeated_scans,
        "coverage": coverage,
        "avg_decision_time_ms": avg_decision_time_ms,
        "avg_latency": avg_latency,
        "dom_counts": dict(dom_counts)
    }

def calc_stats(values):
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, math.sqrt(variance)

def run_experiment():
    print("--- APSS Explicit Coverage Budget Experiment ---")
    
    filepath = "data/tsrd/tsrd_50k.csv"
    adapter = TSRDAdapter()
    all_pdws = adapter.convert_to_pdws(filepath)
    all_pdws = sorted(all_pdws, key=lambda p: p.toa)
    
    num_bands = 100
    start_freq = 0.0
    band_width = 170.0
    window_size = 5000.0
    
    segments = [
        all_pdws[0:10000],
        all_pdws[10000:20000],
        all_pdws[20000:30000],
        all_pdws[30000:40000],
        all_pdws[40000:50000]
    ]
    
    configs = [
        ("A: 20%", 0.20),
        ("B: 30%", 0.30),
        ("C: 40%", 0.40),
        ("D: 50%", 0.50),
    ]
    
    all_results = {c[0]: [] for c in configs}
    
    for i, seg_pdws in enumerate(segments):
        print(f"\n========================================")
        print(f"Segment {i+1}/5 (PDWs {i*10000} to {(i+1)*10000})")
        print(f"========================================")
        
        for cfg_name, target_cov in configs:
            res = evaluate_apss(cfg_name, target_cov, seg_pdws, num_bands, band_width, start_freq, window_size)
            all_results[cfg_name].append(res)
            
            print(f"\nConfig: {res['name']}")
            print(f"Total Scans: {res['total_scans']}, Hits: {res['total_hits']}, Det Rate: {res['detection_rate']:.2f}%")
            print(f"Coverage: {res['coverage']:.1f}%, Unique Bands: {res['unique_bands_scanned']}/100, Repeated Scans: {res['repeated_scans']}")
            print(f"Avg Latency: {res['avg_latency']:.1f}, Avg Decision Time: {res['avg_decision_time_ms']:.4f} ms")
            print(f"Score Component Dominance: {res['dom_counts']}")

    print("\n\n--- Overall Statistical Summary (Across 5 Segments) ---")
    
    for cfg_name in [c[0] for c in configs]:
        det_rates = [r['detection_rate'] for r in all_results[cfg_name]]
        cov_rates = [r['coverage'] for r in all_results[cfg_name]]
        latencies = [r['avg_latency'] for r in all_results[cfg_name]]
        dec_times = [r['avg_decision_time_ms'] for r in all_results[cfg_name]]
        unique_counts = [r['unique_bands_scanned'] for r in all_results[cfg_name]]
        
        det_m, det_s = calc_stats(det_rates)
        cov_m, cov_s = calc_stats(cov_rates)
        lat_m, lat_s = calc_stats(latencies)
        dec_m, dec_s = calc_stats(dec_times)
        uni_m, uni_s = calc_stats(unique_counts)
        
        print(f"\n{cfg_name}:")
        print(f"Detection Rate : {det_m:6.2f}% (std: {det_s:5.2f}%)")
        print(f"Coverage       : {cov_m:6.2f}% (std: {cov_s:5.2f}%)")
        print(f"Unique Bands   : {uni_m:6.1f} (std: {uni_s:5.2f})")
        print(f"Avg Latency    : {lat_m:8.1f} (std: {lat_s:8.1f})")
        print(f"Avg Dec. Time  : {dec_m:8.4f} (std: {dec_s:8.4f})")

if __name__ == "__main__":
    run_experiment()
