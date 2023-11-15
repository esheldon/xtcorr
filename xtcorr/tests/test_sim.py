import numpy as np
from ..sim import simulate_streams


def test_sims_smoke():
    seed = 9898
    rng = np.random.RandomState(seed)

    _ = simulate_streams(rng)
