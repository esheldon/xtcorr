import numpy as np


def simulate_streams(
    rng,
    tstart=0.0,
    tend=10000.0,
    nx=100,
    ny=100,
    rate=1.0,
    delay=1.0,
    delay_sigma=0.1,
    dx=2.0,
    dx_sigma=0.1,
    dy=2.0,
    dy_sigma=0.1,
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

    data1 = make_data(
        rng=rng, num=ntot1, tstart=tstart, tend=tend, nx=nx, ny=ny,
    )
    data2 = make_data(
        rng=rng, num=ntot2, tstart=tstart, tend=tend, nx=nx, ny=ny,
    )

    # "source" is uniform in location and time, but with offsets
    # for other stream
    delays = rng.normal(loc=delay, scale=delay_sigma, size=ndata)
    dxvals = rng.normal(loc=dx, scale=dx_sigma, size=ndata)
    dyvals = rng.normal(loc=dy, scale=dy_sigma, size=ndata)

    data2['time'][:ndata] = data1['time'][:ndata] + delays
    data2['x'][:ndata] = data1['x'][:ndata] + dxvals
    data2['y'][:ndata] = data1['y'][:ndata] + dyvals

    data1.sort(order='time')
    data2.sort(order='time')

    return data1, data2


def make_data(rng, num, tstart, tend, nx, ny):
    """
    make data, initialized with random draws
    """
    dt = [('time', 'f4'), ('x', 'f4'), ('y', 'f4')]
    data = np.zeros(num, dtype=dt)
    data['time'] = rng.uniform(low=tstart, high=tend, size=num)
    data['x'] = rng.uniform(low=0, high=nx-1, size=num)
    data['y'] = rng.uniform(low=0, high=ny-1, size=num)

    return data
