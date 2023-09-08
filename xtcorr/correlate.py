import numpy as np
from numba import njit


@njit
def correlate(times1, times2, dtlow, dthigh, binsize):
    """
    cross correlate two time streams

    times1: array
        The times for events in stream 1
    times2: array
        The times for events in stream 2
    dtlow: float
        Lowest time difference of stream 2 relative to stream 1
        Events from stream 2 will be within the window [dtlow, dthigh]
    dthigh: float
        Highest time difference of stream 2 relative to stream 1
        Events from stream 2 will be within the window [dtlow, dthigh]

    Returns
    -------
    dts: array
        The time differences corresponding to the histogram
    hist: array
        Histogram counting events with time differences given in dts
    """
    window = (dthigh - dtlow)

    nbin = int(window / binsize)
    hist = np.zeros(nbin)
    dts = np.linspace(start=dtlow, stop=dthigh, num=nbin)

    n2 = times2.size

    for t1 in times1:
        tlow = t1 + dtlow
        thigh = t1 + dthigh

        # times2[i2low] strictly >= val
        i2low = bisect_left(times2, tlow, 0, n2-1)

        # times2[i2high] strictly <= val
        i2high = bisect_right(times2, thigh, i2low, n2-1)
        # i2high = bisect_right(times2, thigh, 0, n2-1)

        for i2 in range(i2low, i2high+1):
            t2 = times2[i2]

            binnum = int((t2 - tlow) / binsize)
            if binnum < nbin:
                hist[binnum] += 1

    return dts, hist


@njit
def bisect_left(data, x, lo, hi):
    """
    binary search with returned index `i` satisfies a[i-1] < v <= a[i]
    """

    while lo < hi:
        mid = (lo + hi) // 2
        if data[mid] < x:
            lo = mid + 1
        else:
            hi = mid

    return lo


@njit
def bisect_right(data, x, lo, hi):
    """
    binary search with returned index `i` satisfies a[i-1] <= v < a[i]
    """

    while lo < hi:
        mid = (lo + hi) // 2
        if x < data[mid]:
            hi = mid
        else:
            lo = mid + 1

    return lo
