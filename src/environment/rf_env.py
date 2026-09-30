class RFEnvironment:
    """Simulates an RF environment for testing APSS schedulers."""
    def __init__(self, num_bands: int, seed: int = 42):
        self.num_bands = num_bands
        self.seed = seed
        self.time = 0.0
        
        # Simple simulated active bands for V1
        import random
        self.rng = random.Random(self.seed)
        self.active_bands = set(self.rng.sample(range(self.num_bands), k=max(1, self.num_bands // 4)))

    def step(self, delta_t: float):
        """Advances simulation time. Occasionally shifts active bands."""
        self.time += delta_t
        if self.rng.random() < 0.1:  # 10% chance to change activity
            new_active = self.rng.randint(0, self.num_bands - 1)
            old_active = self.rng.choice(list(self.active_bands)) if self.active_bands else -1
            if old_active != -1:
                self.active_bands.remove(old_active)
            self.active_bands.add(new_active)

    def scan(self, band_id: int, dwell_time: float) -> bool:
        """Simulates scanning a band. Returns True if activity is detected."""
        return band_id in self.active_bands
