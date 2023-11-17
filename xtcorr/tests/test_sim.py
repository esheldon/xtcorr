import os
import tempfile
import numpy as np
from ..sim import simulate_streams
from ..util import make_sim_meta, save_sim, load_sim


def test_sim_smoke():
    rng = np.random.default_rng(9898)
    _ = simulate_streams(rng=rng)


def test_sim_readwrite():
    seed = 1929
    integration_time = 10000
    rng = np.random.default_rng(seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        sim_fname = os.path.join(tmpdir, 'sim.fits')

        sim_data = simulate_streams(rng=rng)

        meta = make_sim_meta(
            seed=seed,
            integration_time=integration_time,
        )
        save_sim(
            outfile=sim_fname,
            output=sim_data,
            meta=meta,
        )

        indata, inmeta = load_sim(sim_fname)

        for key in sim_data:
            for name in sim_data[key].dtype.names:
                assert np.all(sim_data[key] == indata[key])

        for name in meta.dtype.names:
            assert np.all(meta[name] == inmeta[name])
