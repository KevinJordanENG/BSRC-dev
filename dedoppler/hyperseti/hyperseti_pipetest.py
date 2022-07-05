# testfile for runtime and performance of hyperseti pipeline

import numpy as np
import time
import logging
from astropy import units as u

from hyperseti import dedoppler
from hyperseti import hitsearch
from hyperseti import find_et
from hyperseti.plotting import imshow_waterfall, imshow_dedopp

file = '/mnt_home/kjordan/JuliaWork/single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5'

def whole_pipeline():
    config = {
        'preprocess': {
            'sk_flag': False,
            'normalize': False,
        },
        'sk_flag': {
            'n_sigma': 3,
        },
        'dedoppler': {
            'boxcar_mode': 'sum',
            'kernel': 'dedoppler',
            'max_dd': 8.0,
            'min_dd': 0.0001,
            'apply_smearing_corr': False,
            'beam_id': 0
        },
        'hitsearch': {
            'threshold': 3,
            'min_fdistance': 100
        },
        'pipeline': {
            'n_boxcar': 4,
            'merge_boxcar_trials': True,
            'n_blank': 2
        }
    }
    
    tic = time.perf_counter()
    dframe = find_et(file, config, gulp_size=2**18, filename_out='/mnt_home/kjordan/JuliaWork/hyper_hits.csv')
    toc = time.perf_counter()
    times = toc - tic
    print(times)

whole_pipeline()