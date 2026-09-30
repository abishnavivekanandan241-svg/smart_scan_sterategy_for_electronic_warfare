import random
from typing import List
from src.environment.pdw import PDW

class PDWGenerator:
    """Generates synthetic 5-D PDWs for testing the APSS prototype."""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        
    def generate_stable_emitter(self, start_time: float, duration: float, pri: float, cf: float, pw: float, aoa: float, amp: float) -> List[PDW]:
        """Stable emitter with constant PRI, CF, PW, AoA, Amp."""
        pdws = []
        t = start_time
        while t < start_time + duration:
            # add small jitter
            t_jitter = t + self.rng.gauss(0, pri * 0.01)
            pdws.append(PDW(
                toa=t_jitter,
                centre_frequency=cf + self.rng.gauss(0, cf * 0.0001),
                pulse_width=pw + self.rng.gauss(0, pw * 0.05),
                aoa=aoa + self.rng.gauss(0, 1.0),
                amplitude=amp + self.rng.gauss(0, 0.5)
            ))
            t += pri
        return pdws
        
    def generate_burst_emitter(self, start_time: float, cf: float, num_pulses: int = 10, pri: float = 0.001) -> List[PDW]:
        """A burst of pulses from a single emitter."""
        return self.generate_stable_emitter(start_time, num_pulses * pri, pri, cf, 1e-6, 45.0, -50.0)
        
    def generate_frequency_agile_emitter(self, start_time: float, duration: float, pri: float, cf_list: List[float]) -> List[PDW]:
        """Emitter that hops between different centre frequencies."""
        pdws = []
        t = start_time
        while t < start_time + duration:
            cf = self.rng.choice(cf_list)
            pdws.append(PDW(
                toa=t,
                centre_frequency=cf,
                pulse_width=2e-6,
                aoa=90.0,
                amplitude=-40.0
            ))
            t += pri
        return pdws
        
    def generate_sudden_emitter(self, start_time: float, cf: float) -> List[PDW]:
        """An emitter that appears suddenly."""
        return self.generate_burst_emitter(start_time, cf, num_pulses=5, pri=0.002)
        
    def combine_scenarios(self, scenarios: List[List[PDW]]) -> List[PDW]:
        """Combine and sort PDWs by ToA."""
        all_pdws = []
        for s in scenarios:
            all_pdws.extend(s)
        all_pdws.sort(key=lambda p: p.toa)
        return all_pdws
