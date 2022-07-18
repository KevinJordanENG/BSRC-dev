# Survey and Benchmarking Dedoppler Pipelines

The work presented in this document provides a survey and analysis of dedoppler based SETI software pipelines. 

These pipeleines make use of a dedoppler search algorithm to identify narrowband technosignatures exhibiting doppler drift. Additionally, the pipelines perform the file handling, data preprocessing, hit-search for identifying potential signals of interest, and generation of output data files for these potential signals of interest.

The purpose of this document is to provide familiarity and a comparison of the capabilities, limitations, and performance between these pipelines. This includes a review of the interfacing options of these tools & experience using them, an analysis of the algorithms used, benchmarking and performance comparisons, and verification of output similarity.

The three SETI software pipelines surveyed were turboSETI, hyperSETI, and SETIcore. Their working repositories can be found below.

### turboSETI

The workhorse and most frequently used. CPU or GPU supported implementations.

Codebase: <https://github.com/UCBerkeleySETI/turbo_seti>

### hyperSETI

*In beta:* A GPU specific implementation.

Codebase: <https://github.com/UCBerkeleySETI/hyperseti>

### SETIcore

High-performance GPU & low-level code rework of turboSETI.

Codebase: <https://github.com/lacker/seticore>

## Interface and User Experience

Fundamental to the utility of software is its interface and the challenges (or lack thereof) that the end user experiences. In this section, the ease of use and interface of each pipeline were explored. The facets explored include the interface options, installation, and the documentation available for each codebase.

### turboSETI

###### Interface Options

There were two interface options available for turboSETI: a command line utility and as a Python package.

**Command Line Utility**: This interfacing option allowed a straightforward and streamlied option to run the entire turboSETI pipeline. The usage was simple with many optional arguemnts. Additionally, the command line utility offered a well built "help" menu with the passing of the `-h` flag as shown below.

```
kjordan@blpc1:~$ turboSETI -h
usage: turboSETI [-h] [-v] [-M MAX_DRIFT] [-m MIN_DRIFT] [-s SNR] [-o OUT_DIR]
                 [-l LOG_LEVEL] [-c COARSE_CHANS] [-n N_COARSE_CHAN]
                 [-p N_PARALLEL] [-b FLAG_PROGRESS_BAR] [-g FLAG_GPU]
                 [-P FLAG_PROFILE] [-S FLAG_SINGLE_PRECISION]
                 [-a FLAG_APPEND_OUTPUT]
                 [filename]

turboSETI doppler drift narrowband search utility version 2.0.18.

positional arguments:
  filename              Name of filename to open (h5 or fil)

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show the turbo_seti and blimpy versions and exit
  -M MAX_DRIFT, --max_drift MAX_DRIFT
                        Set the maximum drift rate threshold. Unit: Hz/sec.
                        Default: 10.0
  -m MIN_DRIFT, --min_drift MIN_DRIFT
                        Set the minimum drift rate threshold. Unit: Hz/sec.
                        Default: 0.0
  -s SNR, --snr SNR     Set the minimum SNR threshold. Default: 25.0
  -o OUT_DIR, --out_dir OUT_DIR
                        Location for output files. Default: local dir.
  -l LOG_LEVEL, --loglevel LOG_LEVEL
                        Specify log level (info, debug, warning)
  -c COARSE_CHANS, --coarse_chans COARSE_CHANS
                        Comma separated string list of coarse channels to
                        analyze.
  -n N_COARSE_CHAN, --n_coarse_chan N_COARSE_CHAN
                        Number of coarse channels to use.
  -p N_PARALLEL, --n_parallel N_PARALLEL
                        Number of dask partitions to run in parallel. Default
                        to 1 (dask not in use)
  -b FLAG_PROGRESS_BAR, --progress_bar FLAG_PROGRESS_BAR
                        Use a progress bar with dask? (y/n)
  -g FLAG_GPU, --gpu FLAG_GPU
                        Compute on the GPU? (y/n)
  -P FLAG_PROFILE, --profile FLAG_PROFILE
                        Profile execution? (y/n)
  -S FLAG_SINGLE_PRECISION, --single_precision FLAG_SINGLE_PRECISION
                        Use single precision (float32)? (y/n)
  -a FLAG_APPEND_OUTPUT, --append_output FLAG_APPEND_OUTPUT
                        Append output DAT & LOG files? (y/n)
```

Some of the more useful flags were to optionally run turboSETI on a GPU if one was available in the computer system. This `-g` flag allowed a significant performance increase as discussed in the Benchmarking section below. Other optional arguements allowed parameterization of how to process the input data, which increased flexibility for the end user.

**Python Package**: The use of turboSETI as a Python package was also straightforward. The whole pipeline was easily ran through a simple import, object instantiation, and function call as shown in a simplified form below.

```
# import needed package
from turbo_seti.find_doppler.find_doppler import FindDoppler
# instantiate FindDoppler
fdop = FindDoppler(datafile=my_HDF5, max_drift=8, snr=25, ...)
# call function to run pipeline
fdop.search()
```

Similar to the command line utility, the `FindDoppler` object had optional arguments that allowed flexibility to the end user of how to process thi input data.

However, it was found that there was not much modularity within the main code blocks (I/O, dedoppler, hit-search) and that these functions were not easily called independently. While this simplifies its usage for those interested only in the results of running the whole pipeline, it made the analysis of the individual algorithms performance more challenging as discussed in the Algorithm Analysis section below. 

###### Install

turboSETI has numerous dependencies. A virtual environment could be created to manage this, or the packages could be added individually. Many of these packages are standard and likely already installed for data processing and astronomy allpications using Python. The full list is as follows.

- Python 3.7+
- astropy
- numpy
- blimpy 2.0.34+ (Breakthrough Listen I/O Methods for Python: <https://github.com/UCBerkeleySETI/blimpy>)
- pandas
- toolz
- fsspec
- dask
- dask[bag]
- numba
- cloudpickle
- cupy (NVIDIA GPU mode only)

All data processing was performed on the Breakthrough Listen servers. As all dependencies and command line utility were already installed on these machines, the installation procedures were not tested. Given the above information, the required time and ease is left to the reader to infer based on their individual computing system and experience.

###### Documentation & Codebase

As turboSETI is the workhorse getting the most usage out of the three pipelines examined, the documentation was the most developed and complete. The README provided a fairly thourough walkthrough of installation, details on how to run the pipeline both as command line and as a python package, and some demo output results. This enabled quick startup to the usage of this software.

The GitHub repository also included a link to complete documentation at <https://turbo-seti.readthedocs.io>. This was very thourough defining functions, their parameters, and descriptions. Info was available for the dedoppler search, data handling, file writing, needed kernels, helper functions, and hit-search. While complete, information about how to use the main functions independently while maintaining desired performance was lacking. This was likeley due to above mentioned general usage case where the user is interested in running the whole pipeline to examine resulting output.

As the codebase was almost entireley Python, the scripts were easy to understand and easily digestible.

### hyperSETI

###### Interface Options

There were two interface options available for turboSETI: a command line utility and as a Python package. Due to version specific dependencies, a conda virtual environment was necessary to run hyperSETI. Further information about this is presented below in the Install section.

**Command Line Utility**: This interfacing option allowed a straightforward and streamlied option to run the entire hyperSETI pipeline. The usage was simple with many optional arguemnts. Additionally, the command line utility offered a well built "help" menu with the passing of the `-h` flag as shown below.

```
(hyperseti) kjordan@blpc1:~$ findET -h
usage: findET [-h] [--output_csv_path OUTPUT_CSV_PATH] [--gulp_size GULP_SIZE] [--max_drift_rate MAX_DRIFT_RATE]
              [--min_drift_rate MIN_DRIFT_RATE] [--snr_threshold SNR_THRESHOLD] [--num_boxcars NUM_BOXCARS]
              [--kernel {dedoppler,kurtosis,ddsk}] [--gpu_id GPU_ID] [--group_level {debug,info,warning}]
              [--debug_list DEBUG_LIST [DEBUG_LIST ...]] [--noskflag] [--nonormalize] [--nosmearcorr]
              [--nomergeboxcar] [--logfile LOGFILE]
              input_path

Make waterfall plots from a single file.

positional arguments:
  input_path            Path of input file.

optional arguments:
  -h, --help            show this help message and exit
  --output_csv_path OUTPUT_CSV_PATH, -o OUTPUT_CSV_PATH
                        Output path of CSV file. Default: ./hits.csv.
  --gulp_size GULP_SIZE, -z GULP_SIZE
                        Number of channels 2^N to process at once (e.g. Number of fine channels in a coarse
                        channel). Defaults to 18 (i.e. 2^18=262144).
  --max_drift_rate MAX_DRIFT_RATE, -M MAX_DRIFT_RATE
                        Maximum doppler drift in Hz/s for searching. Default: 4.0.
  --min_drift_rate MIN_DRIFT_RATE, -m MIN_DRIFT_RATE
                        Minimum doppler drift in Hz/s for searching. Default: 0.001.
  --snr_threshold SNR_THRESHOLD, -s SNR_THRESHOLD
                        Minimum SNR value for searching. Default: 30.0.
  --num_boxcars NUM_BOXCARS, -b NUM_BOXCARS
                        Number of boxcar trials to do, width 2^N e.g. trials=(1,2,4,8,16). Default: 1.
  --kernel {dedoppler,kurtosis,ddsk}, -k {dedoppler,kurtosis,ddsk}
                        Kernel to be used by the dedoppler module. Default: dedoppler.
  --gpu_id GPU_ID, -g GPU_ID
                        ID of GPU device. Default: 0.
  --group_level {debug,info,warning}, -l {debug,info,warning}
                        Level for all functions that are not being debugged. Default: info.
  --debug_list DEBUG_LIST [DEBUG_LIST ...], -d DEBUG_LIST [DEBUG_LIST ...]
                        List of logger names to use level=logbook.DEBUG. Default: nil.
  --noskflag, -F        Do NOT apply spectral kurtosis flagging when normalizing data.
  --nonormalize, -N     Do NOT normalize input data.
  --nosmearcorr, -S     Do NOT apply doppler smearing correction.
  --nomergeboxcar, -X   Do NOT merge boxcar trials.
  --logfile LOGFILE, -L LOGFILE
                        Name of logfile to write to
```

As with turboSETI above, there were many optional arguments that allowed a more precise control of search parameters. Different than turboSETI however, hyperSETI is natively GPU based and therefore offers a higher base-level performance.

**Python Package**: The use of hyperSETI as a python package was straightforward and sipmple. Unlike turboSETI, the individual code blocks were easily called independently making hyperSETI modular. This allowed insight into the individual algorithms performance. Some of the main functions were the dedoppler search only, the hit-search only, the whole pipeline minus I/O, and the whole pipeline including file reading as shown below.

```
import numpy as np
from astropy import units as u
from hyperseti import dedoppler
from hyperseti import hitsearch
from hyperseti import run_pipeline
from hyperseti import find_et
from hyperseti.io import from_h5

# run dedoppler only
dedopp, md = dedoppler(darr, max_dd=8.0, min_dd=0.0001)
# run hit-search only
hits = hitsearch(dedopp, threshold=100, min_fdistance=10)
# run whole pipeline minus I/O
run_pipeline(darr, config)
# run whole pipeline from file
dframe = find_et(my_HDF5, config)
```

Above, note that `run_pipeleine()` and `find_et()` require a configuration argument of a dictionary object. This was simple to implement and contained options for many of the desired parameters needed to process data appropriately.

The dedoppler function above also supports multiple optional arguments allowing the same flexibility as the command line utility.

###### Install

hyperSETI does have dependencies, though fewer than turboSETI. What makes the installation more challenging is the specific versions needed for each package. Therefore, a conda virtual environment was used as is recommended. Some of the critical dependencies are listed below.

- Python 3.7+
- pandas
- astropy
- rapids=22.04
- cupy (part of rapids)
- cusignal (part of rapids)
- blimpy
- hdf5plugin

All data processing was performed on the Breakthrough Listen servers. As all dependencies and command line utility were already installed on these machines as well as the conda virtual environment setup, the installation procedures were not tested. Given the above information, the required time and ease is left to the reader to infer based on their individual computing system and experience.

###### Documentation & Codebase

The documentation provided for hyperSETI was severely lacking. A brief README was included giving a small example script for the main four functions discussed above, and a few installation tips. As the codebase is still in beta, this was understandable but less than ideal. This increased the time needed to sucessfully run the software. There was no readthedocs detailing functions provided, and code commenting was minimal.

At a minimum the codebase was entireley Python, and the scripts easy to understand and easily digestible. Additionally, the implementation of each of the main algorithm blocks as discrete functions allowed clarity in their functioning.

### SETIcore

###### Interface Options

**Command Line**: Given the C++ and CUDA implementation, SETIcore only supported a command line interface either as a command line utility or compiled program. The compiled program was more limited in options, however very simple in usage allowing quick processing. The command line utility provided more flexibility and a few optional arguments allowing the data to processed to the users desired specifications. With the optional `-h` flag, these optional arguments were displayed in a "help" menu as shown below.

```
kjordan@blpc1:~$ seticore -h
usage: seticore [input]
seticore version: 0.0.4
seticore options:
  -h [ --help ]                    produce help message
  --input arg                      alternate way of setting the input .h5 file
  --output arg                     the output .dat file. if not provided, uses 
                                   the input filename but replaces its suffix 
                                   with .dat
  -M [ --max_drift ] arg (=10)     maximum drift in Hz/sec
  -m [ --min_drift ] arg (=0.0001) minimum drift in Hz/sec
  -s [ --snr ] arg (=25)           minimum SNR to report a hit
```

###### Install

Install instructions were provided for the compiled program version of SETIcore. There were limited dependencies for SETIcore, with a simple command needed to install them. 

`sudo apt-get install cmake libboost-all-dev libhdf5-dev ninja-build pkg-config`

A Python environment was used to build the needed tools.

`pip install meson`

Given the nature of C++ programs, all other needed header and code files were contained within the main SETIcore repository. The repository was cloned, submodules updated, and the make scripts ran.

```
git submodule init
git submodule update
meson setup build
cd build
meson compile
```

From there, the SETIcore pipeline program could be called on the desired HDF5 file.

`./seticore /path/to/my_file.h5`

All data processing was performed on the Breakthrough Listen servers. As the command line utility was already installed on these machines, the installation procedures were not tested. Given the above information, the required time and ease is left to the reader to infer based on their individual computing system and experience.

###### Documentation & Codebase

Documentation for SETIcore was lacking, and included only a brief README detailing installation and some troubleshooting tips. No readthedocs provided for individual functions and code files. However, all code was very well commented providing insight into the function parameter, algorithm function, and external linkage.

The implementation being in C++ and CUDA, the codebase was noticably obfuscated compared to the other two pipelines' Python implementation. Using CUDA directly allows the manual control of how the algorithm executes on the GPU and increases performance. However, to those not familiar with CUDA the source code files can be fairly cryptic. As SETIcore is a compiled program, utilizing any specific function's source code and header files independently of the main pipeline proved difficult.

## Algorithm Analysis

Within these doppler drift SETI search pipelines, there are two main algorithm blocks that perform the heavy lifting: the dedoppler algorithm, and the hit-search algorithm. While the file handling and report generation are integral components of the pipelines, the proportional runtime and computational complexity of these are very low comparative to the dedoppler and hit-search blocks. Below, the implementations used in each of the pipeleines were explored.

### turboSETI

###### Dedoppler

turboSETI makes use of the Taylor-Tree algorithm. Originally devised by Taylor (1974), the algorithm was intended to be applied to dispered radio emissions. Dispersed and doppler shifted narrowband radio signals exhibit similar characteristics. The main differences are the quadratic nature of the signal curve for dispersed signals and linear nature of doppler drifting signals, as well as the inverse relations of frequency and time to the signal charactereistics. Therefore, mathematically the problem of dedispersion and dedoppler are able to use the same algorithmic solution with minor modifications to account for above mentioned differences.

The Taylor-Tree algorithm takes the naturally O(N * N * N) computational complexity of the problem, and through regularization reduces the complexity to O(N * N * log N). The regularization allows shared usage of trial drift rates and reduces redundant calculations (Taylor 1974). This provides improved theoretical performance as compared to the base level Brute-Force algorithm.

As turboSETI included options to run on both CPU and GPU, there were two implementations of the Taylor-Tree algorithm respectively.
- **CPU**: Implementation utilizes a Python kernel based on Numba just-in-time compilation. Numba translates Python to optimized machine code at runtime maximizing potential performance. While the high-level nature of Python usually reduces performance, the use of a numba kernel allowed for maximized algorithm performance. Kernel script can be found at <https://github.com/UCBerkeleySETI/turbo_seti/blob/master/turbo_seti/find_doppler/kernels/_taylor_tree/_core_numba.py>.
- **GPU**: Implementation utilizes a Python kernel script using cupy. cupy is an array library that utilizes CUDA Toolkit libraries and allow the running of the Taylor-Tree algorithm on the GPU. This GPU implementation made use of the improvements in performace possible from parallelization of the Taylor-Tree algorithm. While the use of cupy is streamlined and simple, all functions and GPU usage were handled automatically. While user friendly, options for configuring how the algorithm is implemented to run on the GPU was lacking. Kernel script can be found at <https://github.com/UCBerkeleySETI/turbo_seti/blob/master/turbo_seti/find_doppler/kernels/_taylor_tree/_core_cuda.py>.

###### Hit-Search

The hit-search algorithm implemented by turboSETI is a two-step iterative process. The first step involves iteration through each given spectral resolution [freq_start:freq_end] and identifying hits above a certain signal-to-noise-ratio (SNR). The second step involves iterating through all of the identified hits from the first step and determining the "top" hit between nearby frequency channels. The main Python script containing both steps of iterations can be found at <https://github.com/UCBerkeleySETI/turbo_seti/blob/master/turbo_seti/find_doppler/find_doppler.py>. The function named `hitsearch()` found at line 602 handles the first step's iteration, and the function named `tophitsearch()` found at line 687 performs the second step's iteration. The complexity of the entire algorithm was O(N * log N) given the linear search nature of the first step and the reduced dataset of only a few candidate hits in the second step.
- **1st Step**: The identification of hits is performed by subtracting the given median and dividing that result by the given standar deviation. Any resulting SNR above the set threshold is kept for the second step. As the inital step of iteration through each frequency channel is computationally larger than the second step, an optional GPU kernel is included in addition to the CPU implementation. Unlike the dedoppler algorithm above, this kernel was implemented directly CUDA. This provided performance benefit and more direct control of GPU execution, while having the drawback of being less easily understandable code for those unfamiliar. This CUDA kernel script can be found at <https://github.com/UCBerkeleySETI/turbo_seti/blob/master/turbo_seti/find_doppler/kernels/_hitsearch/kernels.cu>.
- **2nd Step**: Using the maximum drift rate and resolution specified, the second step first determines the maximum distance or number of bins apart two overlapping hits can be. Then, a window with dimensions determined by the maximum distance is calculated. The hits found in the first step are then iterated through to identify those with the largest SNR within this window of nearby frequency channels. This second step eliminates redundancy in reporting of hits in adjacent bins. As these windows have been downsampled from the full dataset, turboSETI includes only a CPU implementation of this part of the hit-search algorithm.

### hyperSETI

###### Dedoppler

The dedoppler algorithm used by hyperSETI is the Brute-Force approach. This approach does not perform any optimization of the algorithm itself and instead opts for the natural O(N * N * N) computational complexity.

While the algorithm implemented by hyperSETI has the lowest theoretical performace, the pipeline is natively GPU and relies on this parallelization to acheive better performace. The implementation uses a Python kernel script using cupy similar to turboSETI. Unlike turboSETI however, the kernel has been developed in CUDA passing this directly to the cupy RawKernel object. This allowed the manipulation and specification of GPU execution while providing easy interfacing with the rest of the Python implemented pipeline. Kernel script can be found at <https://github.com/UCBerkeleySETI/hyperseti/blob/master/hyperseti/kernels/dedoppler.py>.

###### Hit-Search

The hit-search algorithm implemented in hyperSETI uses a two-step process. The first step uses a maxima finding function. The second step involves iterating through all of the identified hits from the first step and determining the "top" hit between nearby frequency channels. The main Python script for the hit-search algorithm can be found at <https://github.com/UCBerkeleySETI/hyperseti/blob/master/hyperseti/hits.py>. The function named `hitsearch()` found at line 135 handles the first step's maxima finding, and the function named `merge_hits()` found at line 52 performs the second step's iteration. As the internal functioning of the maxima finding function `argrelmax()` imported from cusignal could only be speculated, an inferrence of theoretical computational complexity was not possible. However, as in turboSETI above the second step operates on a reduced dataset of only found hits and can therefore be assumed to be of approximately O(log N) in coplexity. Therefore, the entire algorithms computational complexity's worst case should be O(N * log N).
- **1st Step**: The initial finding of hits over the specified SNR is handled entirely by `argrelmax()` imported from cusignal. The relative maxima is computed, a mask of data greater than the SNR threshold is computed, and any relative maxima above this theshold are returned. This maxima finding removes the possibility of returning a plateau as the `argrelmax()` function specifically finds relative maxima. This first step is executed on the GPU with a script that can be found at <https://github.com/UCBerkeleySETI/hyperseti/blob/master/hyperseti/peak.py>.
- **2nd Step**: The second step took identified hits from the first step, identified channels and driftrates within tolerances, and returned the hit with the max SNR. The grouping of channels was perfomed as "boxcars". The boxcar is the same concept as the window calculation performed in turboSETI for maximum possible drift rate and number of frequency bins over. However, unlike turboSETI and SETIcore where the width is set by default to 1, hyperSETI allowed the user to specify the boxcar width. This allowed for greater specificity in searching for particular signal bandwidths. As this second step operates only on the hits found in the first step, the implementation was CPU based.

### SETIcore

###### Dedoppler

SETIcore makes use of the Taylor-Tree algorithm. Discussed in greater detail above in turboSETI's dedoppler algorithm section, the Taylor-Tree provides an improved theoretical performance as compared to the Brute-Force algorithm. This optimized algorithm has O(N * N * log N) computational complexity.

SETIcore, like hyperSETI above, is specifically implemented to run on the GPU. As SETIcore is implemented in C++ and CUDA, the low-level nature of these also provided improved performace. The Taylor-Tree kernel was implemented directly in CUDA. This manual programming of the GPU execution allowed maximum performace. The kernel can be found on lines 15-164 of the CUDA script at <https://github.com/lacker/seticore/blob/master/dedoppler.cu>.

###### Hit-Search

The hit-search algorithm implemented by SETIcore is a two-step iterative process. The first step iterates through the data gathering information about top hit candidates. The second step involves iterating through all of the identified hits from the first step and determining the top hit within the given window. The main CUDA script containing both steps of iterations can be found at <https://github.com/lacker/seticore/blob/master/dedoppler.cu>. The function named `findTopPathSums()` found at line 167 handles the first step, and the second step can be found from line 354 on. Similar to turboSETI above, the complexity of the entire algorithm was O(N * log N) given the linear search nature of the first step and the reduced dataset of only a few candidate hits in the second step. However, as the first step was GPU implemented and the second step CPU implemented, the actual proportional performance was not easily inferred.
- **1st Step**: As SETIcore's implementation was not designed to be modular, the first step operated on data that comes directly from the output of the Taylor-Tree dedoppler algorithm. The direct CUDA implementation of both meant that the data stayed in the GPU device memory from the previous algorithm. From the Taylor-Tree output buffer path sums, the largest path sum along a column was identified.
- **2nd Step**: The second steps was comprised of a few subsections. The data on the device was first copied from the GPU back to the host CPU. Then, the median and standard deviation were calculated. The window of possible frequency bins was determined, and each frequency within this window was iterated through to determine the hit with the highest SNR within the window. The identification of this top hit meant there were no redundant hits reported.

## Benchmarking

Important in any analysis, performance was measured. Although hyperSETI offered modular options to independently call the different algorithmic sections, the general use case was the running of the whole pipeleine from HDF5 datafile input to generated report of potentially interesting signals. Many test runs of each pipeleine were performed with both realistic, large, and multi-channel as well as downsampled, smaller, single coarse channel input datasets. To ensure the most reliable runtime comparisons, all processing was performed on Berkeley SETI Research Center's Breakthrough Listen blpc1 server. This computing system included a TITAN Xp GPU which was used for running all GPU implementations.

###### Single Coarse Channel Input

The initial benchmarking involved collection of runtimes for a simplified input dataset. The dataset chosen was a previously collected observation of the Voyager-1 interstellar space probe's communication signal. Though produced by humans, the extraterrestrial origin of the signal made the observation a perfect dataset to verify proper function of the different pipelines. This single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5 file is frequently used for testing code and can be found at <https://blpd14.ssl.berkeley.edu/voyager_2020/single_coarse_channel/>.

For this given input file, each pipeline was run 10 times. The parameters of maximum and minimum drift rate were set at 8.0 and 0.0001 respectively. The avarage runtime was calulated and reported in the plot shown below.

![Single Channel Benchmark Plot](/home/kjordan/juliaNBs/dedoppler/plot_single_cmd.png)

As expected, the CPU performance was the slowest, with significant improvement by all GPU implementations. The Python GPU implementations displayed comparable runtimes with SETIcores low-level CUDA and C++ implementation demonstrating the expected improvement in performance. While the turboSETI and hyperSETI Python implementations were expected to be slower as compared to SETIcore, what was unexpected was their overall similar runtimes given the significant difference in theoretical computational complexity between the Brute-Force and Taylor-Tree algorithms.

###### Multi-Channel Input

The input was changed to a multi-channel and larger dataset to test the pipelines in more true to use conditions. The dataset used was a general obsearvation dataset frequently used in performance testing provided by Kevin Lacker. This dataset was specifically lacking in signals exhibiting doppler drift to verify the pipeline's function.

As this input file was much larger, a reduced number of trial runs were performed on the CPU implementation of turboSETI due to it's extended runtime. Two trials of this turboSETI CPU version were run, with 10 trial runs performed with all other GPU pipelines. The parameters of maximum and minimum drift rate were set at 8.0 and 0.0001 respectively. The avarage runtime was calulated and reported in the plot shown below.

![Multi-Channnel Benchmark Plot](/home/kjordan/juliaNBs/dedoppler/plot_multi_cmd.png)

Similar to the single channel reduced dataset above, the CPU implementation was demonstrated to be the least performant. What was different as compared to the smaller dataset was the comparative decrease in performace of hyperSETI as compared to turboSETI's GPU implementation.

## Input and Output Comparison Between Pipelines

### Input

As stated above in the Benchmarking section, two different input files were tested. These input files represented two opposing scenarios. One, there was a potentially interesting signal exhibiting doppler drift. The other, there were no signals of interest. If the pipelines were to properly function, the output report file detailing where to find this signal in frequency space should be the same between pipelines, and be empty when not provided doppler drifting signals.

To visualize radio observations, a waterfall plot is often used. This waterfall plot has frequency on the horizontal axis, time along the vertical axis, with signal intensity plotted as a heatmap over these axes.

The waterfall plot of the single coarse channel input data was included below.

![Single Channel Waterfall Input Plot](/home/kjordan/dedoppler/pltw16x65536.png)

The waterfall plot of the multi-channel input data was included below.

![Multi-Channel Waterfall Input Plot](/home/kjordan/dedoppler/pltwsingle_coarse_guppi_0011.png)

### Output Comparison

After two of the test runs of each pipeline, the output report files were saved for comparison. These two files were compared to eachother to verify the consistency between runs of the same pipeline, as well as to the output reports across pipelines. These output report files can be found in their pipeline respective folder at this parent directory <https://github.com/KevinJordanENG/BSRC-dev/tree/master/dedoppler>.

The output report file types were not consistent across pipelines with turboSETI and SETIcore outputting .dat files, and hyperSETI .csv files. Also, the fields of these reports were not consistent with turboSETI and SETIcore again matching format and hyperSETI providing less values. The metadata from the input file was better maintained and transferred to the turboSETI and SETIcore output reports.

Ultimately, there were only two main values of interest: the frequency of the interesting signal, and the signal-to-noise-ratio. While the other output data contained context details about the observation, these two values allowed the user to locate and investigate the potential signal of interest in the dataset.

The reports generated by hyperSETI and SETIcore only included a singular frequency of the signal, turboSETI additionally included range values of frequency start and frequency end. While different in data reporting, these functionally were equivalent, and reported the same interesting signal.

The Voyager-1 signal's frequency was reported by each pipeline as follows:
- **turboSETI**: 8419.542731 MHz
- **hyperSETI**: 8419.542730785906 MHz
- **SETIcore**: 8419.542731 MHz

Minus rounding errors, these pipelines were demonstrated to provide equal results identifying frequency of the doppler drifting signal.

The SNR for the above frequencies were reported by each pipeline as follows:
- **turboSETI**: 192.893814
- **hyperSETI**: 261.9039611816406
- **SETIcore**: 192.940872

While all higher than the SNR threshold of 25, there was inconsistency in these associated SNR values across pipelines. While the cause of this can only be speculated, the difference in the way the hit-search algorithm was implemented between the similar iterative turboSETI and SETIcore and the different maxima finding hyperSETI likely acconted for these results.

Using the given frequency window provided in the output report of turboSETI, a waterfall plot of the interesting signal was included below.

![Output Waterfall Plot turboSETI](/home/kjordan/juliaNBs/dedoppler/turboseti/pltwsingle_coarse_guppi_0011_hit1.png)

Voyager-1's signal can be clearly seen, demonstrating the proper function of the pipeline.

As hyperSETI and SETIcore provided only a central frequency, a small window of +/- 200 Hz about this cental frequency was used to generate the below waterfall plot.

![Output Waterfall Plot hyper/core](/home/kjordan/dedoppler/seticore/pltwsingle_coarse_core_out.png)

Again, Voyager-1's signal can be clearly seen, demonstrating the proper function of the pipeline.

As expected, there were no meaningful results reported for the pipeline runs given the input file with no interesting signals present.

###### Oddities

SETIcore and hyperSETI produced empty report files when given the dataset lacking doppler drifting signals. However, turboSETI's report contained lmany isted hits. This was likely resulted due to the way turboSETI preprocessed input data and the physical instrumentation used in collecting the radio observation. This was suggested by the 3 MHz cadence of these reported hits. When the frequencies associated with these reported hits were investigated, there was found to be no signal present. A waterfall plot showing this was included below.

![Empty turboSETI Waterfall Output](/home/kjordan/dedoppler/turboseti/pltw16x65536_empty.png)

hyperSETI reported two additional signal of interest hits as compared to turboSETI and hyperSETI. These were determined to be the side two data communication sidebands of the main carrier signal found above by turboSETI and SETIcore. This demonstrated that hyperSETI's hit-search algorithm was of greater sensitivity and different in implementation than the other pipelines. As these two sideband signals are relatively close to the carrier signal's frequency, it is possible that these were eliminated as duplicate hits in neighboring frequency bins through the second iterative step of both turboSETI and SETIcore. A waterfall plot of one of these sidebands was included below.

![Sideband Signal hyperSETI](/home/kjordan/dedoppler/hyperseti/pltwsingle_coarse_hyper_out.png)

## Conclusions

Through this work benefits, drawbacks, and challenges were made clear. The relative performancees, ease of use, and resulting outputs were considered holistically in the hopes to elucidate the best options and strategies in the implementation of a doppler drifting signal SETI pipeline.

With performed testing, it was found that there was between a 12 and 24 factor of improved performance between the lowest and highest performant pipelines. The use of GPUs and largely parallel processing of the main algorithm sections was the largest contributing factor in increased performance. The differences in turboSETI's CPU and GPU implementation well supported this, as all other parts of the pipeline were the same. A average factor of 5 in spreedup by swithcing to GPU processing was observed as compared to the CPU implementation.

By and large, the benefits of using low-level and close to hardware languages such as C++ and CUDA as compared to higher level Python scripts was well demonstrated. As both turboSETI's GPU implementaion and SETIcore's implementation used the same Taylor-Tree dedoppler algorithm and iterative hit-search, the language of implementation was considered to be the fundamental difference leading to changes in performance.

## Future Work

To build on the benchmark metrics above, an investigation and benchmarking of the runtimes and performace of the individual algorithm blocks would be a beneficial endeavor. This information can then be used in the development and optimization of the most needing sections of the pipeline.

A further goal would be to look at taking the best functioning algorithm blocks, and implementing a comparable pipeline in Julia to maintian the best qualities of the high-level and easy to understand Python pipelines, and the low-level performance benefits of SETIcore.

*July 13th, 2022 - Kevin Jordan and Max Hawkins with support from Dave MacMahon and Daniel Czech*