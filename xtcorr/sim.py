import numpy as np


def simulate_streams(
    rng,
    tstart=0.0,
    tend=10000.0,
    rate=1.0,
    delay=1.0,
    delay_sigma=0.1,
    background_rate=10.0,
):
    """
    Simulate streams of data with real pairs and background

    Parameters
    ----------
    tstart: float
        Start time
    tend: float
        End time
    rate: float
        Rate of signal in same units as tstart and tend
    background_rate: float
        Rate of background in same units as tstart and tend

    Returns
    --------
    data1: array
        times for events in stream 1
    data2: array
        times for events in stream 1
    """

    dt = tend - tstart
    ndata = rng.poisson(rate * dt)
    nback1 = rng.poisson(background_rate * dt)
    nback2 = rng.poisson(background_rate * dt)

    ntot1 = ndata + nback1
    ntot2 = ndata + nback2

    times1 = np.zeros(ntot1)
    times2 = np.zeros(ntot2)

    times = rng.uniform(low=tstart, high=tend, size=ndata)
    delays = rng.normal(loc=delay, scale=delay_sigma, size=ndata)

    times1[:ndata] = times
    times2[:ndata] = times
    times2[:ndata] += delays

    times1[ndata:] = rng.uniform(low=tstart, high=tend, size=nback1)
    times2[ndata:] = rng.uniform(low=tstart, high=tend, size=nback2)

    times1.sort()
    times2.sort()
    return times1, times2
