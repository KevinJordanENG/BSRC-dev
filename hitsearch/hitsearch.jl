# Instantiate / setup needed Julia packages for generating
# dedoppler shifted dataset for hitsearch development.
# Original DopplerDriftSearch package from David MacMahon: davidm@astro.berkeley.edu

using Pkg
Pkg.activate(@__DIR__)
Pkg.add(url="https://github.com/david-macmahon/DopplerDriftSearch.jl.git")
Pkg.instantiate()

using Plots
using CUDA
using FFTW
using HDF5
using Downloads
using LinearAlgebra
using BenchmarkTools
using DopplerDriftSearch

# Turn off legends in plots by default
default(legend=false)
# Disable scalar indexing to prevent use of inefficient access patterns
CUDA.allowscalar(false)

# read in file
file2open = "/home/kjordan/JuliaWork/downsamp.h5"
h5file = h5open(file2open)

# extract matrix values
spect = h5file["data"][:,1,:]

# uncomment to see original spectogram
# heatmap(spect', yflip=true)

# produce working dataset for hitsearch development
rates = 0:-0.1:-8
Nr = length(rates)
fdmat = intfdr(spect, rates)

# uncomment to see dedoppler output freq drift rate matrix
heatmap(fdmat', yflip=true)

# Hitsearch algorithm development. 2 step iterative approach
# step1: calculate median & standard deviation then find all with SNR above thresh
# setp2: iterate through all potential duplicates within max drift possible window

