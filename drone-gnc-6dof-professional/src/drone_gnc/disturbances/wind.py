import numpy as np

class WindDisturbance:
    def __init__(self, sigma=0.10, seed=7): self.rng=np.random.default_rng(seed); self.sigma=sigma
    def acceleration(self): return self.rng.normal(0.0,self.sigma,3)
