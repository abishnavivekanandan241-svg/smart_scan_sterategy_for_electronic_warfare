import time
from collections import defaultdict
from typing import Dict, Any

from src.preprocessing.tsrd_adapter import TSRDAdapter
from src.preprocessing.pdw_processor import PDWPreprocessor
from src.state.band_state import BandState
from src.schedulers.apss import APSScheduler
from src.schedulers.round_robin import RoundRobinScheduler
from src.schedulers.random_scheduler import RandomScheduler

def evaluate_scheduler(name: str, scheduler_cls, scheduler_kwargs, pdws, num_bands: int, band_width: float, start_freq: float, window_size: float):
    processor = PDWPreprocessor(num_bands=num_bands, band_width=band_width, start_freq=start_freq)
    states = [BandState(band_id=i) for i in range(num_bands)]
    scheduler = scheduler_cls(**scheduler_kwargs)
    
    total_scans = 0
    total_hits = 0
    hits_per_band = defaultdict(int)
    scans_per_band = defaultdict(int)
    total_decision_time_ns = 0
    
    # To calculate detection latency (time from pulse existence to actual scan)
    # Map band_id to a list of unprocessed pulse ToAs
    pending_pulses = defaultdict(list)
    latency_records = []
    selected_bands_sequence = []
    
    for start_t, end_t, batch_pdws in processor.batch_by_time_window(pdws, window_size=window_size):
        # Time passes for all bands
        for s in states:
            s.update_time(end_t)
            
        # Register new pulses into the pending queue for latency calculation
        for p in batch_pdws:
            b_id = processor.get_band_id(p.centre_frequency)
            if 0 <= b_id < num_bands:
                pending_pulses[b_id].append(p.toa)
        
        # Scheduler selects exactly ONE band to scan
        t0 = time.perf_counter_ns()
        selected_band = scheduler.select_next(states)
        t1 = time.perf_counter_ns()
        total_decision_time_ns += (t1 - t0)
        
        if selected_band == -1:
            continue
            
        selected_bands_sequence.append(selected_band)
            
        # Simulate scan: Look at reality in this window
        band_stats = processor.process_batch(batch_pdws)
        stats = band_stats.get(selected_band, {"pulse_count": 0, "activity": 0.0})
        
        is_hit = stats.get("pulse_count", 0) > 0
        
        # Calculate latency for all pending pulses in the scanned band
        # Since we scanned it at `end_t`, the latency is `end_t - pulse_toa`
        if pending_pulses[selected_band]:
            for p_toa in pending_pulses[selected_band]:
                latency_records.append(end_t - p_toa)
            pending_pulses[selected_band].clear()
            
        # Update the state of the *selected* band using the simulated observation
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
        "total_scans": total_scans,
        "total_hits": total_hits,
        "detection_rate": detection_rate,
        "unique_bands_scanned": unique_bands_scanned,
        "coverage": coverage,
        "avg_decision_time_ms": avg_decision_time_ms,
        "avg_latency": avg_latency,
        "hits_per_band": dict(hits_per_band),
        "scans_per_band": dict(scans_per_band),
        "selected_bands_sequence": selected_bands_sequence
    }

def print_results(results: Dict[str, Any]):
    print(f"\n{'='*40}")
    print(f"Results for {results['name']}")
    print(f"{'='*40}")
    print(f"Total Scans            : {results['total_scans']}")
    print(f"Simulated Hits         : {results['total_hits']}")
    print(f"Detection Rate         : {results['detection_rate']:.2f}%")
    print(f"Unique Bands Scanned   : {results['unique_bands_scanned']} / 100")
    print(f"Coverage               : {results['coverage']:.1f}%")
    print(f"Avg Decision Time      : {results['avg_decision_time_ms']:.4f} ms")
    if results['avg_latency'] > 0:
        print(f"Avg Detection Latency  : {results['avg_latency']:.1f} time units")
    else:
        print(f"Avg Detection Latency  : N/A")
        
    if "APSS" in results['name']:
        print(f"\nFirst 20 selected bands: {results['selected_bands_sequence'][:20]}")
        print("Scan distribution (band: count):")
        sorted_scans = sorted(results['scans_per_band'].items(), key=lambda x: x[1], reverse=True)
        for b, count in sorted_scans:
            print(f"  Band {b}: {count} scans")


def run_experiment():
    print("--- REAL APSS Decision Loop Experiment ---")
    print("Loading data/tsrd/tsrd_50k.csv ...")
    
    filepath = "data/tsrd/tsrd_50k.csv"
    adapter = TSRDAdapter()
    pdws = adapter.convert_to_pdws(filepath)
    
    print(f"Loaded {len(pdws)} PDWs.")
    print("Note: Simulated Hit/Miss is derived by checking if the chosen band contains real observations in the temporal window.")
    
    num_bands = 100
    start_freq = 0.0
    band_width = 170.0
    window_size = 5000.0
    
    # Run Round Robin
    rr_results = evaluate_scheduler(
        name="Round Robin Scheduler",
        scheduler_cls=RoundRobinScheduler,
        scheduler_kwargs={},
        pdws=pdws,
        num_bands=num_bands,
        band_width=band_width,
        start_freq=start_freq,
        window_size=window_size
    )
    
    # Run Random
    rand_results = evaluate_scheduler(
        name="Random Scheduler",
        scheduler_cls=RandomScheduler,
        scheduler_kwargs={"seed": 42},
        pdws=pdws,
        num_bands=num_bands,
        band_width=band_width,
        start_freq=start_freq,
        window_size=window_size
    )
    
    # Run APSS
    apss_results = evaluate_scheduler(
        name="APSS (Adaptive Predictive Spectrum Scheduler)",
        scheduler_cls=APSScheduler,
        scheduler_kwargs={}, # Uses default weights
        pdws=pdws,
        num_bands=num_bands,
        band_width=band_width,
        start_freq=start_freq,
        window_size=window_size
    )
    
    print_results(rr_results)
    print_results(rand_results)
    print_results(apss_results)

if __name__ == "__main__":
    run_experiment()
