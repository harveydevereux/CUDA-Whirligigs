## Julia installation:

Download for your OS here https://julialang.org/downloads/

### Example setup tested for Ubuntu 16.04

```bash
$ wget https://julialang-s3.julialang.org/bin/linux/x64/1.5/julia-1.5.3-linux-x86_64.tar.gz
tar -xvf julia-1.5.3-linux-x86_64.tar.gz
```

Then a global command to run Julia can be created by making a symbolic link in /home/[USERNAME]/bin/

```bash
sudo ln -s /PATHTO/julia-1.5.3/bin/julia /home/USERNAME/bin/julia-1.5
```

Assuming the command julia-1.5 is setup to run the Julia REPL from bash, the following will install all
the dependancies

```bash
$ julia-1.5 -e "using Pkg; Pkg.add([\"Plots\",\"DelimitedFiles\",\"LinearAlgebra\",\"ProgressMeter\",\"KernelDensity\",\"LaTeXStrings\",\"Measures\",\"Statistics\",\"StatsBase\",\"LsqFit\",\"ArgParse\", \"BlackBoxOptim\", \"Test\"])"
```

## Software Dependency List

Versions are those used to run the repository, other version may work but have not been tested

- Python 3
  - numpy 1.18.5
  - scipy 1.4.1
  - bayesian-optimization 1.2.0
- Julia 1.5
  - PyCall 1.92.4
  - BlackBoxOptim 0.5.0
  - ArgParse 1.1.0
  - Plots 0.29.9
  - JLD 0.10.0
  - ProgressMeter 0.9.0
  - LsqFit 0.11.0
  - KernelDensity 0.5.1
  - LaTeXStrings 1.1.0
  - StatsBase 0.32.2
- CUDA 
  - nvcc (Cuda compilation tools, release 7.5, V7.5.17)
  - CUDA 10.0
- C++ 
  - C++11 standard compiler
  
## Tested On

### OS

4.15.0-126-generic #129~16.04.1-Ubuntu SMP Tue Nov 24 11:22:40 UTC 2020 x86_64 x86_64 x86_64 GNU/Linux

### GPU

GTX 1080 Ti
Driver 410.104
CUDA 10.0
