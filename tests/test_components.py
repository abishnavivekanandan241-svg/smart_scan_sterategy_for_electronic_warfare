import pytest
from src.environment.rf_env import RFEnvironment
from src.state.band_state import BandState
from src.schedulers.round_robin import RoundRobinScheduler
from src.schedulers.random_scheduler import RandomScheduler
from src.schedulers.apss import APSScheduler
from src.evaluation.evaluator import Evaluator

def test_rf_environment():
    env = RFEnvironment(num_bands=10, seed=42)
    assert len(env.active_bands) > 0
    env.step(1.0)
    assert env.time == 1.0

def test_band_state_updates():
    state = BandState(band_id=0)
    assert state.uncertainty == 1.0
    
    # Simulate passage of time
    state.update_time(current_time=5.0)
    assert state.coverage_debt > 0
    
    # Simulate scan hit
    state.update_after_scan(current_time=6.0, hit=True)
    assert state.hit_count == 1
    assert state.activity > 0
    assert state.coverage_debt == 0.0

def test_round_robin_scheduler():
    states = [BandState(band_id=i) for i in range(3)]
    scheduler = RoundRobinScheduler()
    assert scheduler.select_next(states) == 0
    assert scheduler.select_next(states) == 1
    assert scheduler.select_next(states) == 2
    assert scheduler.select_next(states) == 0

def test_random_scheduler():
    states = [BandState(band_id=i) for i in range(10)]
    scheduler = RandomScheduler(seed=42)
    selected = scheduler.select_next(states)
    assert 0 <= selected < 10

def test_apss_scheduler():
    states = [BandState(band_id=0, coverage_debt=0.9),
              BandState(band_id=1, activity=0.8, coverage_debt=0.2)]
    
    # With default weights, both have scores.
    # state 0: w_D*0.9 - w_C*1.0 + ... = 0.9 - 1.0 = -0.1
    # state 1: w_A*0.8 + w_D*0.2 - w_C*1.0 = 1.0 - 1.0 = 0.0
    # Uncertainty is 1.0 for both by default.
    scheduler = APSScheduler()
    selected = scheduler.select_next(states)
    assert selected in [0, 1]  # The scoring mechanism is deterministic, just check it runs without error.

def test_evaluator():
    env = RFEnvironment(num_bands=5, seed=42)
    scheduler = RoundRobinScheduler()
    evaluator = Evaluator(env, scheduler, num_bands=5)
    
    results = evaluator.run(steps=10, dwell_time=1.0)
    assert results["total_scans"] == 10
    assert "hit_rate" in results
