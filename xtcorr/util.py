from .constants import DETNAMES, PAIRS


def save_sim(outfile, output, meta):
    import fitsio
    print('writing:', outfile)
    with fitsio.FITS(outfile, 'rw', clobber=True) as fits:
        for detname, det_output in output.items():
            fits.write(det_output, extname=detname)
        fits.write(meta, extname='meta')


def load_sim(infile):
    import fitsio
    print('reading:', infile)

    output = {}
    with fitsio.FITS(infile) as fits:
        for name in DETNAMES:
            # numba doesn't do non native byte order
            output[name] = _byteswap(fits[name].read())
        meta = fits['meta'].read()

    return output, meta


def _byteswap(arr):
    arr.byteswap(inplace=True)
    arr = arr.newbyteorder('=')
    return arr


def save_corr(outfile, pair_results):
    import fitsio

    print('writing:', outfile)
    with fitsio.FITS(outfile, 'rw', clobber=True) as fits:
        for key, data in pair_results.items():
            fits.write(data, extname=key)


def load_corr(infile):
    import fitsio

    data = {}

    print('reading:', infile)
    with fitsio.FITS(infile) as fits:
        for pair in PAIRS:
            name = pair[0] + pair[1]
            data[name] = fits[name].read()

    return data


def pack_corr_results(dt, hist):
    import numpy as np

    dtype = [('dt', 'f4'), ('num', 'i8')]
    output = np.zeros(dt.size, dtype=dtype)

    output['dt'] = dt
    output['num'] = hist
    return output


def make_sim_meta(seed, integration_time):
    import numpy as np

    dtype = [
        ('seed', 'i8'),
        ('integration_time', 'f8'),
    ]
    meta = np.zeros(1, dtype=dtype)
    meta['seed'] = seed
    meta['integration_time'] = integration_time

    return meta
