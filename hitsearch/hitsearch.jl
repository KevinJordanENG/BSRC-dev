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
using Statistics
using DopplerDriftSearch

# Turn off legends in plots by default
default(legend=false)
# Disable scalar indexing to prevent use of inefficient access patterns
CUDA.allowscalar(false)

# read in file
file2open = "/home/kjordan/juliaNBs/hitsearch/downsamp.h5"
h5file = h5open(file2open)

# extract matrix values
spect = h5file["data"][:,1,:]

# uncomment to see original spectogram
#heatmap(spect', yflip=true)

# produce working dataset for hitsearch development
min_drift = -0.01
max_drift = -8
rates = min_drift:-0.01:max_drift
Nr = length(rates)
fdmat = intfdr(spect, rates)

# uncomment to see dedoppler output freq drift rate matrix
#heatmap(fdmat', yflip=true)

# Hitsearch algorithm development. 2 step iterative approach

# step1: calculate median & standard deviation then find all with SNR above thresh

# stats calc
std_dev = std(fdmat)
mdian = median(fdmat)

# create arrays for needed hit identification
mask = zeros((size(fdmat))[1], (size(fdmat))[2])
freqs_1d_array = zeros((size(fdmat))[1])

# calc window size for not counting duplicate hits
foff = attrs(h5file["data"])["foff"]
tsamp = attrs(h5file["data"])["tsamp"]
num_timesteps = (size(spect))[2]
drift_rate_resolu = (foff * 1e6) / (num_timesteps * tsamp)
window_size = 2 * ceil(abs(max_drift) / abs(drift_rate_resolu))

# iterate thru to find all matrix values above SNR thresh
for i in 1:(size(fdmat))[1] # for each freq chan
    for j in 1:(size(fdmat))[2] #for each trial drift rate bin
        if (fdmat[i, j]-mdian)/std_dev > 25
            mask[i, j] = (fdmat[i, j]-mdian)/std_dev
            freqs_1d_array[i] = 1
            print("hit! SNR: ")
            print(mask[i, j])
            print(", drift rate: ")
            print(-j/100)
            print(", freq index: ")
            println(i)
        end
    end
end

#=for k in 10:10:22
    println("it executed")
end=#

# setp2: iterate through all potential duplicates within max drift possible window

