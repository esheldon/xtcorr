import os
import tempfile
import numpy as np
from ..sim import simulate_streams
from ..correlate import correlate_by_pairs
from ..util import make_sim_meta, save_sim, load_sim, save_corr


def test_sim_smoke():
    rng = np.random.default_rng(9898)
    sim_data = simulate_streams(rng=rng)
    _ = correlate_by_pairs(
        data=sim_data,
        dtlow=-2,
        dthigh=2,
        tbinsize=0.02,
    )


def test_sim_write_smoke():
    seed = 8812
    integration_time = 10000
    rng = np.random.default_rng(seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        sim_fname = os.path.join(tmpdir, 'sim.fits')
        corr_fname = os.path.join(tmpdir, 'corr.fits')

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

        indata, meta = load_sim(sim_fname)
        pair_results = correlate_by_pairs(
            data=indata,
            dtlow=-2,
            dthigh=2,
            tbinsize=0.02,
        )

        save_corr(outfile=corr_fname, pair_results=pair_results)
