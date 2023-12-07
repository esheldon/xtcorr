# TODO maybe take rate as dnum/ds/dlam

class FlatSpectrum(object):
    def __init__(self, rate, lam_low, lam_high, name=None):
        """
        Parameters
        ----------
        rate: float
            photons per second over range
        """
        self.rate = rate
        self.lam_low = lam_low
        self.lam_high = lam_high
        self.lam_width = lam_high - lam_low
        self.name = name

    def get_num(self, dt, lam_range=None):
        """
        Get total number of photons emitted in time dt within specified
        wavelength range
        """
        if lam_range is None:
            lam_range = (self.lam_low, self.lam_high)

        return self.rate * dt * (lam_range[1] - lam_range[0]) / self.lam_width

    def sample(self, rng, dt, lam_range=None):
        """
        sample from the spectrum
        """

        num = self.get_num(dt, lam_range=lam_range)
        num = np.poisson(num)
        return self.rng.uniform(
            low=self.lam_low,
            high=self.lam_high,
            size=num
        )
