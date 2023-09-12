def save_sim(outfile, data1, data2, meta):
    import fitsio
    print('writing:', outfile)
    with fitsio.FITS(outfile, 'rw', clobber=True) as fits:
        fits.write(data1, extname='data1')
        fits.write(data2, extname='data2')
        fits.write(meta, extname='meta')


def load_sim(infile):
    import fitsio
    print('reading:', infile)
    with fitsio.FITS(infile) as fits:
        data1 = fits['data1'].read()
        data2 = fits['data2'].read()
        meta = fits['meta'].read()

    # numba doesn't do non native byte order
    data1 = _byteswap(data1)
    data2 = _byteswap(data2)

    return data1, data2, meta


def _byteswap(arr):
    arr.byteswap(inplace=True)
    arr = arr.newbyteorder('=')
    return arr


def save_corr(outfile, dt, dy, dx, hist):
    import fitsio

    print(hist.shape)
    print('writing:', outfile)
    with fitsio.FITS(outfile, 'rw', clobber=True) as fits:
        fits.write(dt, extname='dt')
        fits.write(dy, extname='dy')
        fits.write(dx, extname='dx')
        fits.write(hist, extname='hist')


def load_corr(infile):
    import fitsio

    data = {}

    print('reading:', infile)
    with fitsio.FITS(infile) as fits:
        for key in ['dt', 'dy', 'dx', 'hist']:
            data[key] = fits[key].read()

    return data


def make_sim_meta(seed, integration_time, delay, delay_sigma):
    import numpy as np

    dtype = [
        ('seed', 'i8'),
        ('integration_time', 'f8'),
        ('delay', 'f8'),
        ('delay_sigma', 'f8'),
    ]
    meta = np.zeros(1, dtype=dtype)
    meta['seed'] = seed
    meta['integration_time'] = integration_time
    meta['delay'] = delay
    meta['delay_sigma'] = delay_sigma

    return meta
