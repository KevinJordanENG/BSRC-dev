# Takes larger HDF5 format filterbank file and downsamples frequency channels
# Produces HDF5 file with same header and information as original file.
# Useful in algorithm testing, analyses, or creation of more manageable datasets.
# by Kevin Jordan: kevin.jordan.ee@gmail.com 2022

using Plots
using HDF5

# file in and out paths
# set to desired paths
# fileout will create or replace file at specified path
file2open = "/home/kjordan/JuliaWork/single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5"
fileout = "/home/kjordan/JuliaWork/downsamp.h5"

# set freq range to downsample to
# modify to desired output parameters!
fstart = 659754 # freq start index value
new_nchans = 512 # use if downsampling by length
#fstop = 660266 #freq stop index value
freq_range = range(fstart, length=new_nchans)

#open input file
h5file = h5open(file2open, "cw")

# get header info for copying
CLASS = attrs(h5file)["CLASS"]
vERSION = attrs(h5file)["VERSION"]
DIMENSION_LABELS = attrs(h5file["data"])["DIMENSION_LABELS"]
az_start = attrs(h5file["data"])["az_start"]
data_type = attrs(h5file["data"])["data_type"]
fch1 = attrs(h5file["data"])["fch1"]
foff = attrs(h5file["data"])["foff"]
machine_id = attrs(h5file["data"])["machine_id"]
nbits = attrs(h5file["data"])["nbits"]
nchans = attrs(h5file["data"])["nchans"]
nifs = attrs(h5file["data"])["nifs"]
source_name = attrs(h5file["data"])["source_name"]
src_dej = attrs(h5file["data"])["src_dej"]
src_raj = attrs(h5file["data"])["src_raj"]
telescope_id = attrs(h5file["data"])["telescope_id"]
tsamp = attrs(h5file["data"])["tsamp"]
tstart = attrs(h5file["data"])["tstart"]
za_start = attrs(h5file["data"])["za_start"]

# get downsampled data & mask
reduced_dataset = h5file["data"][freq_range, 1:1, :]
reduced_mask = h5file["mask"][freq_range, 1:1, :]

# close and free original .h5 file
close(h5file)

# open new .h5 file to copy attributes
# if file already exists replace it
if isfile(fileout) == true
    rm(fileout, force=true)
end
# write downsampled data & mask to new .h5 file
h5open(fileout, "w") do file
    write(file, "data", reduced_dataset)
    write(file, "mask", reduced_mask)
end

# open to copy attributes to
h5new = h5open(fileout, "cw")

# calculate differences in fch1
new_fch1 = (fstart * foff) + fch1

# copy header info to downsampled file
write_attribute(h5new, "CLASS", CLASS)
write_attribute(h5new, "VERSION", vERSION)
write_attribute(h5new["data"], "DIMENSION_LABELS", DIMENSION_LABELS)
write_attribute(h5new["mask"], "DIMENSION_LABELS", DIMENSION_LABELS)
write_attribute(h5new["data"], "az_start", az_start)
write_attribute(h5new["data"], "data_type", data_type)
write_attribute(h5new["data"], "fch1", new_fch1)
write_attribute(h5new["data"], "foff", foff)
write_attribute(h5new["data"], "machine_id", machine_id)
write_attribute(h5new["data"], "nbits", nbits)
write_attribute(h5new["data"], "nchans", new_nchans)
write_attribute(h5new["data"], "nifs", nifs)
write_attribute(h5new["data"], "source_name", source_name)
write_attribute(h5new["data"], "src_dej", src_dej)
write_attribute(h5new["data"], "src_raj", src_raj)
write_attribute(h5new["data"], "telescope_id", telescope_id)
write_attribute(h5new["data"], "tsamp", tsamp)
write_attribute(h5new["data"], "tstart", tstart)
write_attribute(h5new["data"], "za_start", za_start)

# close complete new downsampled file
close(h5new)