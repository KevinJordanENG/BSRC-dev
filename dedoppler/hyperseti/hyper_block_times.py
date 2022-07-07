import numpy as np
import pandas as pd
import time
import logging
from astropy import units as u

from hyperseti import dedoppler
from hyperseti import hitsearch
from hyperseti.io import from_h5
from hyperseti.plotting import imshow_waterfall, imshow_dedopp

tic1 = time.perf_counter()
darr = from_h5('single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5')
toc1 = time.perf_counter()
t_file_in = toc1 - tic1
print(f"Input & preprocessing time: {t_file_in}s")

tic2 = time.perf_counter()
dedopp, md = dedoppler(darr, max_dd=8.0, min_dd=0.0001)
toc2 = time.perf_counter()
t_dedopp = toc2 - tic2
print(f"dedoppler(): {t_dedopp}s")

tic3 = time.perf_counter()
hits = hitsearch(dedopp, threshold=100, min_fdistance=10)
toc3 = time.perf_counter()
t_hits = toc3 - tic3
print(f"hitsearch(): {t_hits}s")

tic4 = time.perf_counter()
hits.to_csv('hits.csv')
toc4 = time.perf_counter()
t_write_csv = toc4 - tic4
print(f"Hit write output: {t_write_csv}s")