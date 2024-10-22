# CUDA-Whirligigs

### Requirements and Setup Instructions

[REQUIREMENTS.md](https://github.com/harveydevereux/CUDA-Whirligigs/blob/main/REQUIREMENTS.md)

### Troubleshooting

[TROUBLESHOOTING.md](https://github.com/harveydevereux/CUDA-Whirligigs/blob/main/TROUBLESHOOTING.md)


### Data

Data in .csv form for N=50,100,200 datasets split into X, Y, and theta dimensions

- 50
  - [X](https://github.com/harveydevereux/CUDA-Whirligigs/blob/main/Data/50-X.csv)
  - [Y](https://github.com/harveydevereux/CUDA-Whirligigs/blob/main/Data/50-Y.csv)
  - [theta](https://github.com/harveydevereux/CUDA-Whirligigs/blob/main/Data/50-theta.csv)
- 100
  - [X](https://github.com/harveydevereux/CUDA-Whirligigs/blob/main/Data/100-X.csv)
  - [Y](https://github.com/harveydevereux/CUDA-Whirligigs/blob/main/Data/100-Y.csv)
  - [theta](https://github.com/harveydevereux/CUDA-Whirligigs/blob/main/Data/100-theta.csv)
- 200
  - [X](https://github.com/harveydevereux/CUDA-Whirligigs/blob/main/Data/200-X.csv)
  - [Y](https://github.com/harveydevereux/CUDA-Whirligigs/blob/main/Data/200-Y.csv)
  - [theta](https://github.com/harveydevereux/CUDA-Whirligigs/blob/main/Data/200-theta.csv)

### Tests

```bash
$ julia-1.5 -O3 test/density.jl
```

Checks that the c++ and julia (Ultimately [Qhull](http://www.qhull.org/)) local density code agree (within 1e-6) on 1000 particles in a box

### Generating Plots

#### Experimental Data

After setting up Julia and Python (**CUDA is *not* required for analysis of experimental data**) the plots for the experimental data can be generated by

```bash
$ cd Analysis
$ julia-1.5 -O3 analysis.jl
```

#### Model Data

Precomputed model data is provided internally [here](https://github.com/harveydevereux/CUDA-Whirligigs/blob/main/Data/alpha-tau-pd-15.jld) for 10s runtime and externally (due to 1GB file size) [here](https://www.dropbox.com/s/ov1fpm10ogk0xfk/BeetleData.tar.gz?dl=0) for 30s 

The plots can be generated with (**Does *not* require CUDA**).

```bash
$ cd Model\ Analysis
$ julia-1.5 -O3 model-analysis.jl
```

Additionally the precomputed model data can be re-generated (**CUDA is required**)

```bash
$ cd Model\ Analysis
$ julia-1.5 -O3 get-model-data.jl
```

### The Model

The Model can be run with ```./CUDAABP```

For help with options consult

```bash
$ ./CUDAABP --help
Incomplete options
Options should be given in pairs, e.g -N 100
Options are: 
-N                                           (int) Number of particles, DEFAULT 100
-T                                           (float) Max time, DEFAULT = 30.0, [seconds]
-dt                                          (float) Time step, DEFAULT = 1.0/300.0,[seconds]
-mur                                         (float) rotational mobility coefficient, DEFAULT = 0.0001
-a                                           (float) Particle major axis, DEFAULT = 2.0, this defines the length scale
-b                                           (float) Particle minor axis, DEFAULT = 2.0, this defines the length scale
Note: (a,b) = (2,2) implies a unit circle
-k                                           (float) Spring constant, DEFAULT = 10.0
-mu                                          (float) accel = f(v,x,t) + mu * sum(collision_forces(x,t)), DEFAULT = 1.0
-Dr                                          (float) rotational diffusion, DEFAULT = 0.0,[rad]^2[s]^-1
-Dt                                          (float) translational diffusion, DEFAULT = 0.00, [r]^2[s]^-1
-v                                           (float) v0, DEFAULT = 10.0 ,[r][s]^-1
--initial-packing-fraction                   (float) density of random intial condition, DEFAULT = 0.5
--box-length                                 (float) length of periodic box, DEFAULT inf => no box
-tau                                         (float) reorientation coefficient, DEFAULT = 0.0
-alpha                                       (float) reorientation exponent, DEFAULT = 0.0
--save-every                                 (int) save state every --save-every time steps, DEFAULT = 10
--random-seed                                (unsigned long) DEFAULT = 31415926535897
-silent                                      suppress cout DEFAULT = 0 (don't suppress)
```

### Fitting

Single population (N=200)

Run a bayesian optimisation

```bash
$ python3.5 bayes-opt.py
```

All populations (N=50,100,200), with 75% of runs as a random initialisation (exploration), 
running for 24 hours and repeating each function evaluation 3 times with 3 different time based seeds.

```bash
$ julia-1.5 -O3 multi-opt.jl --gamma 0.75 --max-hours 24 --n-repeats 3
```

### Code Map

- CUDA ABP Simulation
  - src/NonInertial.cu (simulation entrypoint, optiion handling etc)
  - include/delaunator.hpp (delaunay tesselations for density)
  - include/NonInertial.cuh (simulation engine)
  - include/density.h (wrapper for delaunator)
- Analysis
  - Analysis/Alphashapes (Julia density implementation)
  - Analysis/analysis.jl (script for all plots)
  - Analysis/analysis-utils.jl (utilities for data analysis)
  - test/density.jl (compare julia and c++ density code)
  - test/delaunator-test.cpp (cpp utility for testing density codes)
