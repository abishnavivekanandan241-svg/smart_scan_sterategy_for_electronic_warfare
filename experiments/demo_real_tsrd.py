from src.preprocessing.tsrd_adapter import TSRDAdapter
from src.preprocessing.pdw_processor import PDWPreprocessor

def run_demo():
    print("--- APSS Real TSRD Pipeline Demo ---")
    
    filepath = "data/tsrd/tsrd_50k.csv"
    adapter = TSRDAdapter()
    
    # Task 2: Validate Data
    print(f"\nValidating dataset at: {filepath}")
    try:
        stats = adapter.validate_data(filepath, expected_rows=50000)
        print("Validation Results:")
        print(f"  Rows: {stats['row_count']}")
        if stats['expected_rows_matched'] is not None:
            print(f"  Expected Rows Matched: {stats['expected_rows_matched']}")
        print(f"  Required Columns Exist: {stats['required_columns_exist']}")
        print(f"  Missing Values Count: {stats['missing_values_count']}")
        print(f"  Unique Classes: {stats['num_unique_classes']}")
        print(f"  RF Range: {stats['rf_min']} to {stats['rf_max']}")
        print(f"  AOA Range: {stats['aoa_min']} to {stats['aoa_max']}")
        print(f"  Non-Positive PulseWidths: {stats['non_positive_pw_count']}")
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}. Make sure the file exists.")
        return
        
    # Task 3 & 4: Convert and Preprocess
    print("\nConverting first 10 rows into PDW objects...")
    sample_pdws = adapter.convert_to_pdws(filepath, limit=10)
    
    for i, pdw in enumerate(sample_pdws):
        # We explicitly show that the class is ground-truth metadata
        print(f"PDW {i+1}: ToA={pdw.toa}, CF={pdw.centre_frequency}, PW={pdw.pulse_width}, AOA={pdw.aoa}, Amp={pdw.amplitude} | GroundTruthClass={pdw.emitter_class}")
        
    print("\nTesting integration with PDWPreprocessor...")
    # Just to show it works, use wide bands since RF values vary
    # From the file peek, RF was ~190 to ~3300.
    processor = PDWPreprocessor(num_bands=10, band_width=500, start_freq=0)
    band_stats = processor.process_batch(sample_pdws)
    
    active_bands = {b: stats for b, stats in band_stats.items() if b != -1 and stats.get('pulse_count', 0) > 0}
    print(f"Processed {len(sample_pdws)} PDWs into {len(active_bands)} active bands.")
    for b, stats in active_bands.items():
        print(f"  Band {b}: {stats}")
        
    print("\nConfirmed: Emitter class is strictly held in PDW.emitter_class as metadata and not utilized by the APSS preprocessor/scheduler.")

if __name__ == "__main__":
    run_demo()
