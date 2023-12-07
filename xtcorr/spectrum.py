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

    def get_num(self, dt):
        """
        Get total number of photons emitted in time dt
        """
        return self.rate * dt

    def get_num_lam_bin(self, lam_low, lam_high, dt):
        """
        Get total number of photons emitted in time dt within specified
        wavelength range
        """
        return self.rate * dt * (lam_high - lam_low) / self.lam_width

    # def sample(self, rng, n):
    #     """
    #     sample from the spectrum
    #     """
    #
    #     return self.rng.uniform(
    #         low=self.lam_low,
    #         high=self.lam_high,
    #         size=n
    #     )
