import pytest
from src.environment.pdw import PDW
from src.state.band_state import BandState
from src.schedulers.apss import APSScheduler

def test_apss_deterministic_selection():
    """Test that APSS selects the band with the highest score deterministically."""
    scheduler = APSScheduler()
    states = [BandState(band_id=i) for i in range(5)]
    
    # Manually configure state to force a specific selection
    # APSS Score is basically D + A + P + N + U + M - C
    
    states[0].activity = 0.8
    states[1].activity = 0.9 # Should be selected if all else equal
    states[2].activity = 0.5
    
    selected_band = scheduler.select_next(states)
    assert selected_band == 1, "APSS should select band 1 with highest score"

def test_apss_score_calculation():
    """Test the score calculation function directly."""
    # Using specific weights to test math
    weights = {
        'w_D': 1.0, 'w_A': 2.0, 'w_P': 1.0, 
        'w_N': 1.0, 'w_U': 0.5, 'w_M': 1.0, 'w_C': 1.0
    }
    scheduler = APSScheduler(weights=weights)
    state = BandState(band_id=0)
    
    # Set values
    state.coverage_debt = 0.5   # 0.5 * 1.0 = 0.5
    state.activity = 0.8        # 0.8 * 2.0 = 1.6
    state.persistence = 0.4     # 0.4 * 1.0 = 0.4
    state.novelty = 0.2         # 0.2 * 1.0 = 0.2
    state.uncertainty = 1.0     # 1.0 * 0.5 = 0.5
    state.miss_risk = 0.1       # 0.1 * 1.0 = 0.1
    state.scan_cost = 1.0       # 1.0 * 1.0 = 1.0 (subtracted)
    
    # Total = 0.5 + 1.6 + 0.4 + 0.2 + 0.5 + 0.1 - 1.0 = 2.3
    score = scheduler.compute_score(state)
    assert pytest.approx(score) == 2.3

def test_hit_miss_state_update():
    """Test state update mechanics after a simulated scan."""
    state = BandState(band_id=10)
    
    # Simulate a hit
    current_time = 1000.0
    stats_hit = {"pulse_count": 5, "activity": 0.8}
    state.update_from_pdw_stats(current_time, stats_hit)
    
    assert state.last_scan_time == 1000.0
    assert state.last_result is True
    assert state.hit_count == 1
    assert state.pulse_count == 5
    assert state.activity == 0.8
    assert state.coverage_debt == 0.0
    
    # Simulate a miss
    current_time = 2000.0
    stats_miss = {"pulse_count": 0, "activity": 0.0}
    state.update_from_pdw_stats(current_time, stats_miss)
    
    assert state.last_scan_time == 2000.0
    assert state.last_result is False
    assert state.miss_count == 1
    assert state.activity == 0.0
    assert state.coverage_debt == 0.0

def test_time_update():
    """Test how passage of time affects state before scan."""
    state = BandState(band_id=5)
    state.last_scan_time = 1000.0
    state.uncertainty = 0.5
    state.coverage_debt = 0.1
    state.activity = 0.8
    state.miss_risk = 0.0
    
    # Update cycle 1
    state.update_time(1010.0)
    
    # Debt should increase by discrete 0.05
    assert pytest.approx(state.coverage_debt) == 0.15
    
    # Uncertainty should increase by discrete 0.02
    assert pytest.approx(state.uncertainty) == 0.52
    
    # Miss risk should increase by discrete (0.05 * eff_activity) = 0.05 * 0.8 = 0.04
    assert pytest.approx(state.miss_risk) == 0.04

def test_discrete_increments_and_caps():
    state = BandState(band_id=1)
    state.coverage_debt = 0.0
    state.uncertainty = 0.0
    
    # Test 1: One update cycle gives +0.05 debt and +0.02 uncertainty
    state.update_time(1.0)
    assert pytest.approx(state.coverage_debt) == 0.05
    assert pytest.approx(state.uncertainty) == 0.02
    
    # Test 3: Ten cycles give coverage_debt = 0.50 from zero (we already did 1 cycle, so 9 more)
    for i in range(9):
        state.update_time(float(i + 2))
    assert pytest.approx(state.coverage_debt) == 0.50
    assert pytest.approx(state.uncertainty) == 0.20
    
    # Test 4 & 5: Twenty or more cycles cap coverage_debt and uncertainty at 1.0
    for i in range(50):
        state.update_time(float(i + 12))
    assert pytest.approx(state.coverage_debt) == 1.0
    assert pytest.approx(state.uncertainty) == 1.0
    
    # Test 6: Scan/reset followed by update cycles starts aging again
    state.update_after_scan(100.0, hit=True)
    assert state.coverage_debt == 0.0
    assert pytest.approx(state.uncertainty) == 0.5 # Halved from 1.0
    
    state.update_time(101.0)
    assert pytest.approx(state.coverage_debt) == 0.05
    assert pytest.approx(state.uncertainty) == 0.52

def test_independent_of_time_value():
    # Test 7: Behavior is independent of time_since_scan
    state1 = BandState(band_id=1)
    state2 = BandState(band_id=2)
    
    state1.last_scan_time = 0.0
    state2.last_scan_time = 0.0
    state1.activity = 1.0
    state2.activity = 1.0
    
    state1.update_time(10.0) # Delta 10
    state2.update_time(10000.0) # Delta 10000
    
    assert state1.coverage_debt == state2.coverage_debt
    assert state1.uncertainty == state2.uncertainty
    assert state1.miss_risk == state2.miss_risk
    assert state1.miss_risk == 0.05 # 0.05 * 1.0 = 0.05

def test_miss_risk_discrete_aging():
    state = BandState(band_id=3)
    state.miss_risk = 0.0
    
    # Test 1: Inactive band does not accumulate meaningful Miss Risk
    state.activity = 0.0
    state.recent_activity = 0.0
    state.update_time(10.0)
    assert state.miss_risk == 0.0
    
    # Test 2: A band with activity=1.0 gains +0.05 Miss Risk per skipped decision cycle
    state.activity = 1.0
    state.update_time(20.0)
    assert pytest.approx(state.miss_risk) == 0.05
    
    # Test 3: Ten skipped cycles produce 0.50 Miss Risk from zero
    state.miss_risk = 0.0
    for i in range(10):
        state.update_time(30.0 + i)
    assert pytest.approx(state.miss_risk) == 0.50
    
    # Test 4: Twenty or more cycles cap Miss Risk at 1.0
    for i in range(20):
        state.update_time(50.0 + i)
    assert pytest.approx(state.miss_risk) == 1.0
    
    # Test 6: Scan/reset correctly clears or updates Miss Risk
    stats_hit = {"pulse_count": 5, "activity": 1.0}
    state.update_from_pdw_stats(100.0, stats_hit)
    assert state.miss_risk == 0.0
