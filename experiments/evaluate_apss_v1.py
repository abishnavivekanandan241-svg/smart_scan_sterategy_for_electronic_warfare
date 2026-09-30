import time
import math
from collections import defaultdict
from typing import Dict, Any, List

from src.preprocessing.tsrd_adapter import TSRDAdapter
from src.preprocessing.pdw_processor import PDWPreprocessor
from src.state.band_state import BandState
from src.schedulers.apss import APSScheduler
from src.schedulers.round_robin import RoundRobinScheduler
from src.schedulers.random_scheduler import RandomScheduler

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

def evaluate_scheduler(name: str, scheduler_cls, scheduler_kwargs, pdws, num_bands: int, band_width: float, start_freq: float, window_size: float, is_apss: bool = False):
    processor = PDWPreprocessor(num_bands=num_bands, band_width=band_width, start_freq=start_freq)
    states = [BandState(band_id=i) for i in range(num_bands)]
    scheduler = scheduler_cls(**scheduler_kwargs)
    
    weights = {'w_D': 1.0, 'w_A': 1.0, 'w_P': 1.0, 'w_N': 1.0, 'w_U': 1.0, 'w_M': 1.0, 'w_C': 1.0}
    
    total_scans = 0
    total_hits = 0
    hits_per_band = defaultdict(int)
    scans_per_band = defaultdict(int)
    total_decision_time_ns = 0
    
    pending_pulses = defaultdict(list)
    latency_records = []
    
    dom_counts = defaultdict(int)
    activity_vals = defaultdict(list)
    scores_vals = defaultdict(list)
    
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
        
        if is_apss:
            comps = get_score_components(selected_state, weights)
            total_score = sum(comps.values())
            dominant_comp = max(comps.items(), key=lambda x: x[1])[0]
            dom_counts[dominant_comp] += 1
            activity_vals[selected_band].append(selected_state.activity)
            scores_vals[selected_band].append(total_score)
            
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
    
    top_bands = []
    if is_apss:
        sorted_bands = sorted(scans_per_band.items(), key=lambda x: x[1], reverse=True)[:10]
        for b, count in sorted_bands:
            h = hits_per_band[b]
            hr = (h / count) * 100
            ma = sum(activity_vals[b]) / len(activity_vals[b]) if activity_vals[b] else 0.0
            ms = sum(scores_vals[b]) / len(scores_vals[b]) if scores_vals[b] else 0.0
            top_bands.append({"band": b, "scans": count, "hits": h, "hit_rate": hr, "mean_act": ma, "mean_score": ms})
            
    return {
        "name": name,
        "total_scans": total_scans,
        "total_hits": total_hits,
        "detection_rate": detection_rate,
        "unique_bands_scanned": unique_bands_scanned,
        "coverage": coverage,
        "avg_decision_time_ms": avg_decision_time_ms,
        "avg_latency": avg_latency,
        "top_bands": top_bands,
        "dom_counts": dict(dom_counts)
    }

def calc_stats(values):
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, math.sqrt(variance)

def run_experiment():
    print("--- APSS V1 Evaluation Experiment ---")
    
    filepath = "data/tsrd/tsrd_50k.csv"
    adapter = TSRDAdapter()
    all_pdws = adapter.convert_to_pdws(filepath)
    all_pdws = sorted(all_pdws, key=lambda p: p.toa)
    
    num_bands = 100
    start_freq = 0.0
    band_width = 170.0
    window_size = 5000.0
    
    # 5 deterministic segments of 10,000 PDWs each
    segments = [
        all_pdws[0:10000],
        all_pdws[10000:20000],
        all_pdws[20000:30000],
        all_pdws[30000:40000],
        all_pdws[40000:50000]
    ]
    
    print("\nData Split Method:")
    print("The 50,000 PDW dataset is chronologically sorted and split into 5 deterministic segments of 10,000 PDWs each.")
    print("Limitation: Since the data is sequentially ordered, different segments represent different slices of time, potentially exhibiting highly variable RF characteristics (bursts of activity, silence, or different emitters) which directly impacts evaluation metrics segment-to-segment.")
    
    all_results = {"RR": [], "Random": [], "APSS": []}
    
    for i, seg_pdws in enumerate(segments):
        print(f"\n========================================")
        print(f"Segment {i+1}/5 (PDWs {i*10000} to {(i+1)*10000})")
        print(f"========================================")
        
        rr_res = evaluate_scheduler("Round Robin", RoundRobinScheduler, {}, seg_pdws, num_bands, band_width, start_freq, window_size)
        rand_res = evaluate_scheduler("Random", RandomScheduler, {"seed": 42+i}, seg_pdws, num_bands, band_width, start_freq, window_size)
        apss_res = evaluate_scheduler("APSS", APSScheduler, {}, seg_pdws, num_bands, band_width, start_freq, window_size, is_apss=True)
        
        all_results["RR"].append(rr_res)
        all_results["Random"].append(rand_res)
        all_results["APSS"].append(apss_res)
        
        for res in [rr_res, rand_res, apss_res]:
            print(f"\nScheduler: {res['name']}")
            print(f"Total Scans: {res['total_scans']}, Hits: {res['total_hits']}, Det Rate: {res['detection_rate']:.2f}%")
            print(f"Coverage: {res['coverage']:.1f}%, Unique Bands: {res['unique_bands_scanned']}/100")
            print(f"Avg Latency: {res['avg_latency']:.1f}, Avg Decision Time: {res['avg_decision_time_ms']:.4f} ms")
            
            if res['name'] == "APSS":
                print("Top 10 Selected Bands:")
                for b in res['top_bands']:
                    print(f"  Band {b['band']:<3}: {b['scans']} scans, {b['hits']} hits, {b['hit_rate']:.1f}% hit rate, Act: {b['mean_act']:.2f}")
                print(f"Score Component Dominance: {res['dom_counts']}")

    print("\n\n--- Overall Statistical Summary (Across 5 Segments) ---")
    
    for key, name in [("RR", "Round Robin"), ("Random", "Random"), ("APSS", "APSS")]:
        det_rates = [r['detection_rate'] for r in all_results[key]]
        cov_rates = [r['coverage'] for r in all_results[key]]
        latencies = [r['avg_latency'] for r in all_results[key]]
        
        det_m, det_s = calc_stats(det_rates)
        cov_m, cov_s = calc_stats(cov_rates)
        lat_m, lat_s = calc_stats(latencies)
        
        print(f"\n{name}:")
        print(f"Detection Rate : {det_m:6.2f}% (std: {det_s:5.2f}%)")
        print(f"Coverage       : {cov_m:6.2f}% (std: {cov_s:5.2f}%)")
        print(f"Avg Latency    : {lat_m:8.1f} (std: {lat_s:8.1f})")

if __name__ == "__main__":
    run_experiment()
