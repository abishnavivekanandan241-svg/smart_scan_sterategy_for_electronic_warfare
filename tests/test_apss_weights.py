import pytest
from src.state.band_state import BandState
from src.schedulers.apss import APSScheduler

def test_weights_affect_score():
    """Verify that changing weights affects the final score calculation."""
    state = BandState(band_id=0)
    state.activity = 0.5
    state.coverage_debt = 1.0
    state.uncertainty = 1.0
    
    # Baseline weights
    scheduler1 = APSScheduler(weights={'w_D': 1.0, 'w_A': 1.0, 'w_P': 1.0, 'w_N': 1.0, 'w_U': 1.0, 'w_M': 1.0, 'w_C': 1.0})
    score1 = scheduler1.compute_score(state)
    
    # Coverage focused weights
    scheduler2 = APSScheduler(weights={'w_D': 5.0, 'w_A': 1.0, 'w_P': 1.0, 'w_N': 1.0, 'w_U': 5.0, 'w_M': 1.0, 'w_C': 1.0})
    score2 = scheduler2.compute_score(state)
    
    assert score2 > score1, "Higher coverage/uncertainty weights should increase the score for this state"

def test_deterministic_selection():
    """Verify that APSS selection is deterministic given the same state and weights."""
    states = [BandState(band_id=i) for i in range(5)]
    states[2].activity = 1.0
    
    scheduler = APSScheduler()
    selected1 = scheduler.select_next(states)
    selected2 = scheduler.select_next(states)
    
    assert selected1 == selected2 == 2, "Selection must be deterministic"

def test_coverage_debt_influences_priority():
    """Verify that coverage debt directly increases priority."""
    scheduler = APSScheduler()
    states = [BandState(band_id=0), BandState(band_id=1)]
    
    # Band 1 has higher coverage debt
    states[0].coverage_debt = 0.1
    states[1].coverage_debt = 0.9
    
    assert scheduler.select_next(states) == 1

def test_uncertainty_influences_priority():
    """Verify that uncertainty directly increases priority."""
    scheduler = APSScheduler()
    states = [BandState(band_id=0), BandState(band_id=1)]
    
    # Band 1 has higher uncertainty
    states[0].uncertainty = 0.2
    states[1].uncertainty = 1.0
    
    assert scheduler.select_next(states) == 1
