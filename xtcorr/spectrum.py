import numpy as np


def make_spectrum(config, name=None):
    if config['spec_type'] == 'flat':
        spec = FlatSpectrum(rate=config['rate'], name=name)
    else:
        raise ValueError(f'bad spec_type: {config["spec_type"]}')
    return spec


class FlatSpectrum(object):
    def __init__(self, rate, name=None):
        """
        Parameters
        ----------
        rate: float
            photons per second per cm^2 per angstrom
        """
        self.rate = rate
        self.name = name

    def get_num(self, dt, lam_min, lam_max, area):
        """
        Get total number of photons emitted in time dt within specified
        wavelength range

        Parameters
        ----------
        dt: float
            Time interval in seconds
        lam_min: float
            Minimum lambda value
        lam_max: float
            Maximum lambda value
        area: float
            Area of detector in m^2
        """
        area_cm = area * 100**2
        return self.rate * dt * area * (lam_max - lam_min)

    def sample(self, rng, dt, lam_min, lam_max, area):
        """
        sample from the spectrum

        Parameters
        ----------
        rng: np.random.default_rng
            Random number generator
        dt: float
            Number of seconds
        lam_min: float
            Minimum lambda value
        lam_max: float
            Maximum lambda value
        area: float
            Area of detector in m^2
        """

        num = self.get_num(dt, lam_min=lam_min, lam_max=lam_max, area=area)
        num = rng.poisson(num)
        return rng.uniform(
            low=lam_min,
            high=lam_max,
            size=num
        )
