import pytest
from src.preprocessing.pdw_processor import PDWPreprocessor
from src.environment.pdw import PDW

def test_temporal_windowing():
    prep = PDWPreprocessor(num_bands=2, band_width=100.0, start_freq=0.0)
    
    # Create PDWs with specific ToAs, unsorted
    pdws = [
        PDW(toa=5.5, centre_frequency=50.0, pulse_width=1e-6, aoa=0.0, amplitude=0.0), # Window 5.0 - 6.0
        PDW(toa=1.2, centre_frequency=50.0, pulse_width=1e-6, aoa=0.0, amplitude=0.0), # Window 1.0 - 2.0
        PDW(toa=1.8, centre_frequency=50.0, pulse_width=1e-6, aoa=0.0, amplitude=0.0), # Window 1.0 - 2.0
        PDW(toa=3.1, centre_frequency=50.0, pulse_width=1e-6, aoa=0.0, amplitude=0.0), # Window 3.0 - 4.0
    ]
    
    # Process with window_size = 1.0
    # Expected windows:
    # 1. Start 1.2, End 2.2 -> [1.2, 1.8]
    # 2. Start 2.2, skipped... next ToA is 3.1
    # Actually based on logic:
    # current_start = 1.2. Window: 1.2 to 2.2. Includes 1.2, 1.8.
    # next pdw = 3.1. Skipped = int((3.1 - 1.2)/1.0) = 1. 1.2 + 1.0 = 2.2.
    # So next window is 2.2 to 3.2. Includes 3.1.
    # next pdw = 5.5. Skipped = int((5.5 - 2.2)/1.0) = 3. 2.2 + 3.0 = 5.2.
    # next window is 5.2 to 6.2. Includes 5.5.
    
    batches = list(prep.batch_by_time_window(pdws, window_size=1.0))
    
    assert len(batches) == 3
    
    w1_start, w1_end, w1_pdws = batches[0]
    assert w1_start == 1.2
    assert w1_end == 2.2
    assert len(w1_pdws) == 2
    assert w1_pdws[0].toa == 1.2
    assert w1_pdws[1].toa == 1.8
    
    w2_start, w2_end, w2_pdws = batches[1]
    assert w2_start == 2.2
    assert w2_end == 3.2
    assert len(w2_pdws) == 1
    assert w2_pdws[0].toa == 3.1
    
    w3_start, w3_end, w3_pdws = batches[2]
    assert w3_start == 5.2
    assert w3_end == 6.2
    assert len(w3_pdws) == 1
    assert w3_pdws[0].toa == 5.5

def test_temporal_windowing_empty():
    prep = PDWPreprocessor(num_bands=2, band_width=100.0, start_freq=0.0)
    batches = list(prep.batch_by_time_window([], window_size=1.0))
    assert len(batches) == 0
