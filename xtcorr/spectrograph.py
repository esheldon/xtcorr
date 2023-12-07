import numpy as np


class RSpecGraph(object):
    def __init__(self, R, lam_min, lam_max):
        self.R = R
        self.lam_min = lam_min
        self.lam_max = lam_max
        self._make_bins()

    def get_binnum(self, lam):
        isscalar = np.isscalar(lam)
        lam = np.array(lam, ndmin=1)
        binnums = np.zeros(lam.size, dtype='i4')
        binnums[:] = -1

        w, = np.where(
            (lam >= self.lam_min)
            &
            (lam <= self.lam_max)
        )
        binnums[w] = np.searchsorted(self.bins['end'], lam)

        if isscalar:
            binnums = binnums[0]

        return binnums

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

            lam += dlam
            if lam > self.lam_max:
                break

        self.bins = np.hstack(bins)
