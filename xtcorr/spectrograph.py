import numpy as np


class RSpecGraph(object):
    def __init__(self, R, lam_min, lam_max):
        self.R = R
        self.lam_min = lam_min
        self.lam_max = lam_max
        self._make_bins()

    def _make_bins(self):
        dtype = [('start', 'f8'), ('end', 'f8')]
        bins = []

        lam = self.lam_min
        while True:
            dlam = lam / self.R

            one = np.zeros(1, dtype=dtype)
            one['start'] = lam
            one['end'] = lam + dlam
            bins.append(one)
            bins.append((lam, lam + dlam))
            lam += dlam

        self.bins = np.hstack(bins)
