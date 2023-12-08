from .constants import DETNAMES, PAIRS


def save_sim(outfile, output, lam_bins, config, meta):
    import fitsio
    print('writing:', outfile)

    config_data = make_config_output(config)

    with fitsio.FITS(outfile, 'rw', clobber=True) as fits:
        for detname, det_output in output.items():
            fits.write(det_output, extname=detname)
        fits.write(lam_bins, extname='lam_bins')
        fits.write(config_data, extname='config')
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


def make_sim_meta(
    seed,
    integration_time,
    theta1,
    theta2,
):
    import numpy as np

    dtype = [
        ('seed', 'i8'),
        ('integration_time', 'f8'),
        ('theta1', 'f8'),
        ('theta2', 'f8'),
    ]
    meta = np.zeros(1, dtype=dtype)
    meta['seed'] = seed
    meta['integration_time'] = integration_time
    meta['theta1'] = theta1
    meta['theta2'] = theta2

    return meta


def make_config_output(config):
    import yaml
    import numpy as np

    cstr = yaml.dump(config)
    dtype = [('config', 'U%d' % len(cstr))]
    config_data = np.zeros(1, dtype=dtype)
    config_data['config'] = cstr
    return config_data
