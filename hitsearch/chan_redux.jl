# Takes larger HDF5 file and downsamples frequency channels
# Useful in algorithm testing. kevin.jordan.ee@gmail.com

using Plots
using HDF5

filepath = "/home/kjordan/JuliaWork/single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5"
h5file = h5open(filepath, "cw")
#size(h5file["data"])
freq_range = range(659754, length=512)
reduced_dataset = h5file["data"][freq_range, 1:1, :]
size(reduced_dataset)
