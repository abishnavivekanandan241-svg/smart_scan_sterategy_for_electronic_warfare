from src.preprocessing.tsrd_adapter import TSRDAdapter
from src.preprocessing.pdw_processor import PDWPreprocessor
from src.state.band_state import BandState

def run_experiment():
    print("--- APSS Real TSRD Temporal Processing Demo ---")
    
    filepath = "data/tsrd/tsrd_50k.csv"
    adapter = TSRDAdapter()
    
    print(f"Loading real TSRD data from {filepath}...")
    pdws = adapter.convert_to_pdws(filepath)
    print(f"Loaded {len(pdws)} PDWs.")
    
    num_bands = 100
    start_freq = 0.0
    band_width = 170.0
    processor = PDWPreprocessor(num_bands=num_bands, band_width=band_width, start_freq=start_freq)
    
    states = [BandState(band_id=i) for i in range(num_bands)]
    
    # Configure temporal window
    # From previous runs, we saw ToA range up to 1097254.5. 
    # Let's use a window size that gives a meaningful progression.
    # We will just print the first 5 windows that have data.
    window_size = 5000.0 
    
    print(f"\nProcessing chronologically with Window Size: {window_size} time units")
    
    windows_shown = 0
    max_windows_to_show = 5
    
    for start_t, end_t, batch_pdws in processor.batch_by_time_window(pdws, window_size=window_size):
        band_stats = processor.process_batch(batch_pdws)
        active_in_batch = {b: s for b, s in band_stats.items() if s.get("pulse_count", 0) > 0}
        
        # Update all states up to end_t
        for b in range(num_bands):
            states[b].update_time(end_t)
            
            if b in active_in_batch:
                states[b].update_from_pdw_stats(end_t, active_in_batch[b])
            else:
                # If no activity in this window, we still pass empty stats to trigger 
                # activity decay and persistence EMA for a missed observation window.
                # However, our V1 update_from_pdw_stats assumes the band was actually "scanned".
                # For this temporal evaluation, we treat all bands as omnisciently monitored 
                # at the end of the window to show how state evolves natively.
                empty_stats = {"pulse_count": 0, "activity": 0.0}
                states[b].update_from_pdw_stats(end_t, empty_stats)
                
        # Only print windows that have activity, to avoid spam
        if len(active_in_batch) > 0:
            print(f"\n--- Temporal Window [{start_t:.1f} to {end_t:.1f}] ---")
            print(f"PDWs in window: {len(batch_pdws)}")
            print(f"Active bands: {list(active_in_batch.keys())}")
            
            print(f"{'Band ID':<8} | {'Pulse Cnt':<10} | {'Activity':<10} | {'Persistence':<12} | {'Novelty':<10}")
            for b in sorted(active_in_batch.keys())[:5]: # Show first 5 active
                s = states[b]
                print(f"{s.band_id:<8} | {s.pulse_count:<10} | {s.activity:<10.2f} | {s.persistence:<12.2f} | {s.novelty:<10.2f}")
                
            windows_shown += 1
            
        if windows_shown >= max_windows_to_show:
            print(f"\n... (stopping after showing {max_windows_to_show} active windows)")
            break
            
    print("\nDemonstration successful: States update incrementally per temporal window.")

if __name__ == "__main__":
    run_experiment()
