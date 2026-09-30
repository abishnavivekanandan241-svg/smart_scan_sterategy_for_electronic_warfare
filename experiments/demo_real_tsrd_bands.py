from src.preprocessing.tsrd_adapter import TSRDAdapter
from src.preprocessing.pdw_processor import PDWPreprocessor
from src.state.band_state import BandState

def run_experiment():
    print("--- APSS Real TSRD Bands Experiment ---")
    
    filepath = "data/tsrd/tsrd_50k.csv"
    adapter = TSRDAdapter()
    
    print(f"Loading real TSRD data from {filepath}...")
    pdws = adapter.convert_to_pdws(filepath)
    print(f"Loaded {len(pdws)} PDWs.")
    
    # Configure the APSS Experimental Grid
    # Note: This is an APSS experiment configuration designed to cover the 
    # observed RF range of the dataset. It does NOT claim that TSRD officially 
    # defines 100 bands of 170 RF units (MHz).
    num_bands = 100
    start_freq = 0.0
    band_width = 170.0
    
    print("\nAPSS Experimental Configuration:")
    print(f"  Bands: {num_bands}")
    print(f"  Start Frequency: {start_freq}")
    print(f"  Band Width: {band_width}")
    print(f"  Total Coverage: {start_freq} to {start_freq + num_bands * band_width} RF units")
    
    # Process PDWs through the configured pipeline
    processor = PDWPreprocessor(num_bands=num_bands, band_width=band_width, start_freq=start_freq)
    band_stats = processor.process_batch(pdws)
    
    # Reporting
    out_of_bounds = band_stats.pop(-1, {}).get("out_of_bounds_count", 0)
    print(f"\nMapped Records: {len(pdws) - out_of_bounds}")
    print(f"Out of Bounds Records: {out_of_bounds}")
    
    total_invalid_pw = sum(stats.get("invalid_pw_count", 0) for stats in band_stats.values())
    print(f"Invalid PulseWidth Count (Ignored from APSS stats): {total_invalid_pw}")
    
    active_bands = {b: s for b, s in band_stats.items() if s["pulse_count"] > 0}
    print(f"\nFound {len(active_bands)} active frequency bands.")
    
    # Show Pulse Count and Activity for Active Bands
    print("\n--- Active Band Activity ---")
    print(f"{'Band ID':<8} | {'Valid Pulses':<13} | {'Invalid PWs':<12} | {'Activity':<8} | {'Avg Amp (dBm)'}")
    for b in sorted(active_bands.keys())[:10]: # Limit to 10 for display
        s = active_bands[b]
        print(f"{b:<8} | {s['pulse_count']:<13} | {s['invalid_pw_count']:<12} | {s['activity']:<8.2f} | {s['avg_amplitude']:.2f}")
    if len(active_bands) > 10:
        print("... (showing first 10 active bands)")
        
    # Update Band State
    print("\n--- Band States After Processing (First 5 Active Bands) ---")
    states = [BandState(band_id=i) for i in range(num_bands)]
    for b, stats in band_stats.items():
        states[b].update_from_pdw_stats(current_time=1.0, stats=stats)
        
    print(f"{'Band ID':<8} | {'State Activity':<15} | {'State Persistence':<18} | {'State Novelty'}")
    shown = 0
    for s in states:
        if s.pulse_count > 0:
            print(f"{s.band_id:<8} | {s.activity:<15.2f} | {s.persistence:<18.2f} | {s.novelty:<13.2f}")
            shown += 1
            if shown >= 5:
                break
                
    print("\nExperiment Successful. Invalid PWs are tracked but do not affect valid state parameters.")

if __name__ == "__main__":
    run_experiment()
