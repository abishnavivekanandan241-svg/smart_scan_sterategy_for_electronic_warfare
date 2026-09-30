import time
from collections import defaultdict
from typing import Dict, Any, List

from src.preprocessing.tsrd_adapter import TSRDAdapter
from src.preprocessing.pdw_processor import PDWPreprocessor
from src.state.band_state import BandState
from src.schedulers.apss import APSScheduler

def evaluate_apss_config(name: str, weights: Dict[str, float], pdws, num_bands: int, band_width: float, start_freq: float, window_size: float):
    processor = PDWPreprocessor(num_bands=num_bands, band_width=band_width, start_freq=start_freq)
    states = [BandState(band_id=i) for i in range(num_bands)]
    scheduler = APSScheduler(weights=weights)
    
    total_scans = 0
    total_hits = 0
    hits_per_band = defaultdict(int)
    scans_per_band = defaultdict(int)
    total_decision_time_ns = 0
    
    pending_pulses = defaultdict(list)
    latency_records = []
    
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
            hits_per_band[selected_band] += 1

    unique_bands_scanned = len([b for b, c in scans_per_band.items() if c > 0])
    coverage = (unique_bands_scanned / num_bands) * 100 if num_bands > 0 else 0
    detection_rate = (total_hits / total_scans) * 100 if total_scans > 0 else 0
    avg_decision_time_ms = (total_decision_time_ns / total_scans) / 1_000_000 if total_scans > 0 else 0
    avg_latency = sum(latency_records) / len(latency_records) if latency_records else 0.0
    
    return {
        "name": name,
        "weights": weights,
        "total_scans": total_scans,
        "total_hits": total_hits,
        "detection_rate": detection_rate,
        "unique_bands_scanned": unique_bands_scanned,
        "coverage": coverage,
        "avg_decision_time_ms": avg_decision_time_ms,
        "avg_latency": avg_latency,
        "hits_per_band": dict(hits_per_band),
        "scans_per_band": dict(scans_per_band)
    }

def run_experiment():
    print("--- APSS Weight Sensitivity Experiment ---")
    
    filepath = "data/tsrd/tsrd_50k.csv"
    adapter = TSRDAdapter()
    pdws = adapter.convert_to_pdws(filepath)
    
    num_bands = 100
    start_freq = 0.0
    band_width = 170.0
    window_size = 5000.0
    
    configs = [
        {
            "name": "A. Current baseline weights",
            "weights": {
                'w_D': 1.0, 'w_A': 1.0, 'w_P': 1.0, 
                'w_N': 1.0, 'w_U': 1.0, 'w_M': 1.0, 'w_C': 1.0
            }
        },
        {
            "name": "B. Coverage-balanced",
            "weights": {
                'w_D': 5.0, 'w_A': 1.0, 'w_P': 1.0, 
                'w_N': 1.0, 'w_U': 5.0, 'w_M': 1.0, 'w_C': 1.0
            }
        },
        {
            "name": "C. Exploration-balanced",
            "weights": {
                'w_D': 8.0, 'w_A': 0.1, 'w_P': 0.1, 
                'w_N': 5.0, 'w_U': 8.0, 'w_M': 0.1, 'w_C': 1.0
            }
        }
    ]
    
    results = []
    for cfg in configs:
        res = evaluate_apss_config(
            name=cfg["name"],
            weights=cfg["weights"],
            pdws=pdws,
            num_bands=num_bands,
            band_width=band_width,
            start_freq=start_freq,
            window_size=window_size
        )
        results.append(res)
        
    for res in results:
        print(f"\n========================================")
        print(f"Results for {res['name']}")
        print(f"Weights: {res['weights']}")
        print(f"========================================")
        print(f"Total Scans            : {res['total_scans']}")
        print(f"Simulated Hits         : {res['total_hits']}")
        print(f"Detection Rate         : {res['detection_rate']:.2f}%")
        print(f"Unique Bands Scanned   : {res['unique_bands_scanned']} / 100")
        print(f"Coverage               : {res['coverage']:.1f}%")
        print(f"Avg Decision Time      : {res['avg_decision_time_ms']:.4f} ms")
        if res['avg_latency'] > 0:
            print(f"Avg Detection Latency  : {res['avg_latency']:.1f} time units")
        else:
            print(f"Avg Detection Latency  : N/A")
            
    print("\n--- Comparison Table ---")
    print(f"{'Configuration':<30} | {'Detection Rate':<15} | {'Coverage':<10} | {'Unique Bands':<15} | {'Avg Latency':<15} | {'Avg Decision Time'}")
    for res in results:
        name = res['name']
        det = f"{res['detection_rate']:.2f}%"
        cov = f"{res['coverage']:.1f}%"
        bands = f"{res['unique_bands_scanned']}/100"
        lat = f"{res['avg_latency']:.1f}"
        dt = f"{res['avg_decision_time_ms']:.4f} ms"
        print(f"{name:<30} | {det:<15} | {cov:<10} | {bands:<15} | {lat:<15} | {dt}")

if __name__ == "__main__":
    run_experiment()
