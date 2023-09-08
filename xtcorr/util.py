def save_sim(outfile, times1, times2, meta):
    import fitsio
    print('writing:', outfile)
    with fitsio.FITS(outfile, 'rw', clobber=True) as fits:
        fits.write(times1, extname='times1')
        fits.write(times2, extname='times2')
        fits.write(meta, extname='meta')


def load_sim(infile):
    import fitsio
    print('reading:', infile)
    with fitsio.FITS(infile) as fits:
        times1 = fits['times1'].read()
        times2 = fits['times2'].read()
        meta = fits['meta'].read()

    return times1, times2, meta


def save_corr(outfile, dts, hist):
    import numpy as np
    import fitsio

    dtype = [('dt', 'f8'), ('hist', 'i8')]
    output = np.zeros(dts.size, dtype=dtype)
    output['dt'] = dts
    output['hist'] = hist

    print('writing:', outfile)
    with fitsio.FITS(outfile, 'rw', clobber=True) as fits:
        fits.write(output, extname='corr')


def load_corr(infile):
    import fitsio

    print('reading:', infile)
    with fitsio.FITS(infile) as fits:
        data = fits['corr'].read()

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
