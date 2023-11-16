import numpy as np
from ..sim import simulate_streams

def test_sim_smoke():
    rng = np.random.default_rng(9898)
    _ = simulate_streams(rng=rng)
