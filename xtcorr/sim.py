"""
for du of 10 GHz, dt is about 0.1 nanoseconds

If we use units of nanoseconds, then dt would be 0.1 (currently working at 0.02 in
arbitrary units)

Then would pick tend to be bigger as well by factor of ~10 to get similar stats
at given wavelength

Would need rates 10 times higher, 10 rather than 1
"""
import numpy as np
from numba import njit
from .correlate import bisect_left, bisect_right
from .constants import DETNAMES


def simulate_streams(
    rng,
    spec1,
    spec2,
    spec_graph,
    tstart=0.0,
    tend=100000.0,
    dt=0.1,
):
    """
    Generate streams for two sources


    Parameters
    ----------
    rng: np.random.default_rng
        The random number generator
    spec1: Spectrum
        A spectrum object for source 1
    spec2: Spectrum
        A spectrum object for source 2
    spec_graph: spectrograph object
        E.g. .spectrgraph.RSpecGraph
    tstart: float
        Start time
    tend: float
        End time
    """
    pass


def simulate_streams_one(
    rng,
    tstart=0.0,
    tend=10000.0,
    rate1=1.0,
    rate2=1.0,
    delta=0.0,
    dt=0.02,
):
    """
    Simulate streams for two sources at a single wavelength

    Parameters
    ----------
    rng: np.random.default_rng
        The random number generator
    tstart: float
        Start time
    tend: float
        End time
    rate1: float
        Rate of photons from source 1
    rate2: float
        Rate of photons from source 2
    delta: float
        Difference delta1 - delta2
    dt: float
        Source photons within this time difference are considered
        as coincicent

    Returns
    --------
    dict with these entries

    data_c: array
        times for events in detector c
    data_d: array
        times for events in detector d
    data_g: array
        times for events in detector g
    data_h: array
        times for events in detector h
    """

    total_time = tend - tstart
    ndata1 = rng.poisson(rate1 * total_time)
    ndata2 = rng.poisson(rate2 * total_time)

    # photons from source 1
    data1 = make_data(rng=rng, num=ndata2, tstart=tstart, tend=tend)
    # photons from source 2
    data2 = make_data(rng=rng, num=ndata1, tstart=tstart, tend=tend)

    data1.sort(order='time')
    data2.sort(order='time')

    # these max at 1/4
    P12_02_13 = (1/8) * (1 + np.cos(delta))
    P12_03_12 = (1/8) * (1 - np.cos(delta))
    print(f'P12_02_13: {P12_02_13} P12_03_12: {P12_03_12}')

    distribute_coincidences(
        rng=rng,
        data1=data1,
        data2=data2,
        dt=dt,
        P12_02_13=P12_02_13,
        P12_03_12=P12_03_12,
    )

    output = {}
    for detector, detname in enumerate(DETNAMES):
        w1, = np.where(data1['detector'] == detector)
        w2, = np.where(data2['detector'] == detector)
        output[detname] = make_output_data(data1[w1], data2[w2])

    return output


@njit
def distribute_coincidences(rng, data1, data2, dt, P12_02_13, P12_03_12):
    """
    Given streams of photons from two sources, distribute to detectors.
    Identify coincidences and distribute appropriately

    rng: np.random.Generator
        e.g. from np.random.default_rng(seed)
    data: data array
        must have fields 'time' and 'detector' and be sorted by time
    dt: float
        Time window will be [-dt/2, dt/2]

    Side Effects
    ------------
    The detector field is filled in randomly
    """

    n2 = data2.size

    for tdata1 in data1:

        tlow = tdata1['time'] - dt/2.0
        thigh = tdata1['time'] + dt/2.0

        i2low = bisect_left(data2['time'], tlow, 0, n2-1)
        i2high = bisect_right(data2['time'], thigh, i2low, n2-1)

        for i2 in range(i2low, i2high):
            tdata2 = data2[i2]

            # found a coincidence, we will assign detectors according
            # to the probabilities
            r = rng.uniform()
            # half of outcomes go to cc/dd/gg/hh
            if r < 0.5:
                # distribute equally to cc, dd, gg, hh
                detector = int(r * 8)
                tdata1['detector'] = detector
                tdata2['detector'] = detector
            else:
                # the rest, from 0.5 to 1, go to these.
                # P12 (cg) = P12 (dh) = (1/8)(1 + cos(δ1 − δ2 ))
                # P12 (ch) = P12 (dg) = (1/8)(1 − cos(δ1 − δ2 ))
                # so we can test r-0.5 against them
                r -= 0.5

                # P12_02_13 and P12_03_12 each max at 1/4, but describes
                # prob for two possible stats. Split between the two
                if r < P12_02_13:
                    tdata1['detector'] = 0
                    tdata2['detector'] = 2
                elif r < P12_02_13 * 2:
                    tdata1['detector'] = 1
                    tdata2['detector'] = 3
                else:
                    r -= P12_02_13 * 2
                    if r < P12_03_12:
                        tdata1['detector'] = 0
                        tdata2['detector'] = 3
                    else:
                        tdata1['detector'] = 1
                        tdata2['detector'] = 2


def make_data(rng, num, tstart, tend):
    """
    make data, initialized with random draws
    """
    dt = [('time', 'f4'), ('detector', 'i2')]
    data = np.zeros(num, dtype=dt)
    data['time'] = rng.uniform(low=tstart, high=tend, size=num)

    # default is distribute randomly to all detectors
    # we will look for coincidences from different sources later
    data['detector'] = rng.integers(0, 4, size=num)

    return data


def make_output_data(d1, d2):
    num = d1.size + d2.size

    dt = [('time', 'f4')]
    data = np.zeros(num, dtype=dt)
    data['time'][:d1.size] = d1['time']
    data['time'][d1.size:] = d2['time']
    data.sort(order='time')
    return data
