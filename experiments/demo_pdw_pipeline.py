from src.environment.pdw_generator import PDWGenerator
from src.preprocessing.pdw_processor import PDWPreprocessor
from src.state.band_state import BandState
from src.schedulers.apss import APSScheduler

def run_demo():
    print("--- APSS PDW Pipeline Demo ---")
    
    # 1. Setup Environment Scenario
    gen = PDWGenerator(seed=123)
    # Scenario: Sudden emitter in band 1 (1.1 GHz) and stable emitter in band 3 (1.3 GHz)
    pdws_stable = gen.generate_stable_emitter(start_time=0.0, duration=0.05, pri=0.001, cf=1.35e9, pw=1e-6, aoa=45, amp=-40)
    pdws_sudden = gen.generate_sudden_emitter(start_time=0.01, cf=1.15e9)
    all_pdws = gen.combine_scenarios([pdws_stable, pdws_sudden])
    
    print(f"Generated {len(all_pdws)} synthetic PDWs in scenario.")
    print(f"Sample PDW: {all_pdws[0]}")
    
    # 2. Preprocess PDWs
    # 5 bands, 100MHz each, starting from 1GHz
    # Band 0: 1.0-1.1 GHz
    # Band 1: 1.1-1.2 GHz
    # Band 2: 1.2-1.3 GHz
    # Band 3: 1.3-1.4 GHz
    # Band 4: 1.4-1.5 GHz
    processor = PDWPreprocessor(num_bands=5, band_width=100e6, start_freq=1e9)
    band_stats = processor.process_batch(all_pdws)
    
    print(f"\nSample Preprocessed Band (Band 1): {band_stats[1]}")
    
    # 3. Initialize and Update State
    states = [BandState(band_id=i) for i in range(5)]
    # Add some dummy coverage debt to simulate time passed
    states[0].coverage_debt = 0.5
    states[2].coverage_debt = 0.8
    states[4].coverage_debt = 0.1
    
    for band_id, stats in band_stats.items():
        states[band_id].update_from_pdw_stats(current_time=0.05, stats=stats)
        
    # 4. APSS Scoring and Selection
    # Slightly higher weight for novelty to prioritize the sudden emitter
    custom_weights = {
        'w_D': 0.5, # Coverage Debt
        'w_A': 1.0, # Activity
        'w_P': 0.5, # Persistence
        'w_N': 2.0, # Novelty (very important)
        'w_U': 0.5, # Uncertainty
        'w_M': 1.0, # Miss Risk
        'w_C': 1.0, # Scan Cost
    }
    scheduler = APSScheduler(weights=custom_weights)
    
    print("\n--- Current Band States & APSS Scores ---")
    print(f"{'Band':<5} | {'Pulse Cnt':<10} | {'Activity':<8} | {'Persist':<8} | {'Novelty':<8} | {'Debt':<5} | {'Score':<8}")
    
    best_band_id = -1
    best_score = -float('inf')
    
    for s in states:
        score = scheduler.compute_score(s)
        if score > best_score:
            best_score = score
            best_band_id = s.band_id
        
        print(f"{s.band_id:<5} | {s.pulse_count:<10} | {s.activity:<8.2f} | {s.persistence:<8.2f} | {s.novelty:<8.2f} | {s.coverage_debt:<5.2f} | {score:<8.2f}")
        
    print(f"\nSelected next band: {best_band_id}")
    
    # Why?
    best_state = states[best_band_id]
    print(f"\nWhy Band {best_band_id} was selected:")
    print(f"w_N * Novelty ({custom_weights['w_N']} * {best_state.novelty:.2f}) = {custom_weights['w_N'] * best_state.novelty:.2f}")
    print(f"w_A * Activity ({custom_weights['w_A']} * {best_state.activity:.2f}) = {custom_weights['w_A'] * best_state.activity:.2f}")
    print(f"w_D * Debt ({custom_weights['w_D']} * {best_state.coverage_debt:.2f}) = {custom_weights['w_D'] * best_state.coverage_debt:.2f}")

if __name__ == "__main__":
    run_demo()
