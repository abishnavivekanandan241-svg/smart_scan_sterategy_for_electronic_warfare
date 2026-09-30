import pytest
from src.environment.pdw import PDW
from src.environment.pdw_generator import PDWGenerator
from src.preprocessing.pdw_processor import PDWPreprocessor
from src.preprocessing.tsrd_adapter import TSRDAdapter
from src.state.band_state import BandState

def test_pdw_creation():
    p = PDW(toa=1.0, centre_frequency=1e9, pulse_width=1e-6, aoa=90.0, amplitude=-40.0)
    assert p.centre_frequency == 1e9
    assert p.amplitude == -40.0

def test_pdw_generator_determinism():
    gen1 = PDWGenerator(seed=42)
    gen2 = PDWGenerator(seed=42)
    pdws1 = gen1.generate_burst_emitter(0.0, 1e9, 10, 0.001)
    pdws2 = gen2.generate_burst_emitter(0.0, 1e9, 10, 0.001)
    assert len(pdws1) == len(pdws2)
    assert pdws1[0].toa == pdws2[0].toa

def test_cf_to_band_mapping():
    prep = PDWPreprocessor(num_bands=10, band_width=100e6, start_freq=1e9)
    assert prep.get_band_id(1e9) == 0
    assert prep.get_band_id(1.05e9) == 0
    assert prep.get_band_id(1.1e9) == 1
    assert prep.get_band_id(0.9e9) == -1 # Out of bounds
    assert prep.get_band_id(2.5e9) == -1 # Out of bounds

def test_pdw_preprocessing_aggregation():
    prep = PDWPreprocessor(num_bands=2, band_width=100e6, start_freq=1e9)
    pdws = [
        PDW(toa=1.0, centre_frequency=1.05e9, pulse_width=1e-6, aoa=10.0, amplitude=-20.0),
        PDW(toa=1.1, centre_frequency=1.05e9, pulse_width=1e-6, aoa=10.0, amplitude=-20.0)
    ]
    results = prep.process_batch(pdws)
    assert results[0]["pulse_count"] == 2
    assert results[0]["avg_amplitude"] == -20.0
    assert results[1]["pulse_count"] == 0

def test_empty_pdw_batch():
    prep = PDWPreprocessor(num_bands=2, band_width=100e6, start_freq=1e9)
    results = prep.process_batch([])
    assert results[0]["pulse_count"] == 0
    assert results[1]["pulse_count"] == 0
    assert results[-1]["out_of_bounds_count"] == 0

def test_frequency_mapping_edge_cases():
    prep = PDWPreprocessor(num_bands=3, band_width=100.0, start_freq=0.0)
    # Band 0: 0.0 - 99.99
    # Band 1: 100.0 - 199.99
    # Band 2: 200.0 - 299.99
    
    # 1. Exactly at start_freq
    assert prep.get_band_id(0.0) == 0
    # 2. Inside a band
    assert prep.get_band_id(150.0) == 1
    # 3. At a band boundary
    assert prep.get_band_id(200.0) == 2
    # 4. Above configured range
    assert prep.get_band_id(300.0) == -1
    # 5. Below configured range
    assert prep.get_band_id(-10.0) == -1

def test_invalid_pulsewidth_handling():
    prep = PDWPreprocessor(num_bands=1, band_width=100.0, start_freq=0.0)
    pdws = [
        PDW(toa=1.0, centre_frequency=50.0, pulse_width=1e-6, aoa=10.0, amplitude=-20.0),
        PDW(toa=1.1, centre_frequency=50.0, pulse_width=0.0, aoa=10.0, amplitude=-20.0), # Invalid
        PDW(toa=1.2, centre_frequency=50.0, pulse_width=-1e-6, aoa=10.0, amplitude=-20.0) # Invalid
    ]
    results = prep.process_batch(pdws)
    
    b0 = results[0]
    assert b0["invalid_pw_count"] == 2
    assert b0["pulse_count"] == 1
    assert b0["avg_pw"] == 1e-6
    assert b0["activity"] == 0.01

def test_band_state_pdw_update():
    state = BandState(band_id=0)
    stats = {
        "pulse_count": 50,
        "activity": 0.5,
        "avg_amplitude": -30.0,
        "avg_pw": 1e-6,
        "avg_aoa": 45.0
    }
    state.update_from_pdw_stats(1.0, stats)
    assert state.pulse_count == 50
    assert state.activity == 0.5
    assert state.novelty == 0.5 # 0.5 - 0.0
    assert state.persistence > 0
    assert state.avg_aoa == 45.0

    # Second update with identical activity
    state.update_from_pdw_stats(2.0, stats)
    assert state.pulse_count == 100
    assert state.novelty == 0.0 # 0.5 - 0.5

def test_tsrd_adapter():
    adapter = TSRDAdapter()
    raw = {"UTCTime": 2.5, "RF": 2e9, "PulseWidth": 2e-6, "AOA": 180.0, "PA": -10.0, "Class": 3}
    pdw = adapter.adapt_record(raw)
    assert pdw.toa == 2.5
    assert pdw.centre_frequency == 2e9
    assert pdw.emitter_class == 3

def test_emitter_scenarios():
    gen = PDWGenerator()
    # Stable
    stable = gen.generate_stable_emitter(0.0, 1.0, 0.1, 1e9, 1e-6, 90.0, -50.0)
    assert len(stable) > 0
    # Agile
    agile = gen.generate_frequency_agile_emitter(0.0, 1.0, 0.1, [1e9, 1.1e9])
    assert len(agile) > 0
    # Sudden
    sudden = gen.generate_sudden_emitter(0.5, 1e9)
    assert len(sudden) == 5
