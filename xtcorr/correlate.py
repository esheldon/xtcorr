"""
TODO:

    - why building up at x/y = 0?
    - why not centered on pixel?
"""
import numpy as np
from numba import njit


@njit
def correlate(
    data1, data2,
    dtlow, dthigh, tbinsize,
    dylow, dyhigh, ybinsize,
    dxlow, dxhigh, xbinsize,
):
    """
    cross correlate two time streams

    data1: array
        The data for events in stream 1
    data2: array
        The data for events in stream 2
    dtlow: float
        Lowest time difference of stream 2 relative to stream 1
        Events from stream 2 will be within the window [dtlow, dthigh]
    dthigh: float
        Highest time difference of stream 2 relative to stream 1
        Events from stream 2 will be within the window [dtlow, dthigh]
    tbinsize: float
        Binsize for time

    Returns
    -------
    result: dict
        The dictionary has entries
        dt: array
            The time differences corresponding to the histogram
        dy: array
            The y differences corresponding to the histogram
        dx: array
            The x differences corresponding to the histogram
        hist: array
            Histogram counting events (ny, nx, nt)
    """
    twindow = (dthigh - dtlow)
    ywindow = (dyhigh - dylow)
    xwindow = (dxhigh - dxlow)

    ntbin = int(twindow / tbinsize)
    nxbin = int(xwindow / xbinsize)
    nybin = int(ywindow / ybinsize)

    hist = np.zeros((nybin, nxbin, ntbin))

    n2 = data2.size

    for tdata1 in data1:

        tlow = tdata1['time'] + dtlow
        thigh = tdata1['time'] + dthigh

        # times2[i2low] strictly >= val
        i2low = bisect_left(data2['time'], tlow, 0, n2-1)

        # times2[i2high] strictly <= val
        i2high = bisect_right(data2['time'], thigh, i2low, n2-1)
        # i2high = bisect_right(times2, thigh, 0, n2-1)

        for i2 in range(i2low, i2high+1):
            tdata2 = data2[i2]

            dt = tdata2['time'] - tdata1['time']
            tbinnum = int((dt - dtlow) / tbinsize)

            # due to int truncation issues negative near zero, we check float
            # value
            if dt > dtlow and tbinnum < ntbin:

                dx = tdata2['x'] - tdata1['x']
                xbinnum = int((dx - dxlow) / xbinsize)

                if dx > dxlow and xbinnum < nxbin:

                    dy = tdata2['y'] - tdata1['y']
                    ybinnum = int((dy - dylow) / ybinsize)

                    if dy > dylow and ybinnum < nybin:
                        hist[ybinnum, xbinnum, tbinnum] += 1

    out_dt = np.linspace(start=dtlow, stop=dthigh, num=ntbin)
    out_dx = np.linspace(start=dxlow, stop=dxhigh, num=nxbin)
    out_dy = np.linspace(start=dylow, stop=dyhigh, num=nybin)

    return out_dt, out_dy, out_dx, hist


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
