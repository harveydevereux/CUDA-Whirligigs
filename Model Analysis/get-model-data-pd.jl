using Plots
using DelimitedFiles
using LinearAlgebra
using ProgressMeter
using LsqFit
using KernelDensity
using Statistics
using LaTeXStrings
using StatsBase
using Measures
using JLD2
using PyCall
stats = pyimport("scipy.stats")
include("../Analysis/AlphaShapes/AlphaShapes.jl")
include("../Analysis/analysis-utils.jl")
# this is usually an "overnighter"
using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "-T"
            arg_type = Float64
			default = 10.			# seconds of simulated data
    end
    return parse_args(s)
end

args = parse_commandline()
@info args

model_seconds = args["T"]

@info "Simulating $model_seconds seconds of model data each iteration"

"""
A wrapper to run the model and collect the trajectories
"""
function Sim(N,alpha,tau,v,mu,mur=1.,dt=1.0/900.0)
    cmd = `../CUDAABP --initial-packing-fraction 0.5 -N $N -T $model_seconds -a 1 -b 1 -alpha $alpha -tau $tau -dt $dt -mu $mu -mur $mur -v $v -Dr 2.34 --random-seed $(Int(floor(time()))) -silent 1`
    run(cmd)
    T = readdlm("trajectories.txt",',')
    #h = Int(floor(size(T,1)/2))
    d_data = T[:,1:2]
    return d_data
end

d = Sim(10,1,1,1,30,1.);

 len = Int(ceil(size(d,1) / 10))

runs = 3
Ps = [200]

n = 15

alpha = collect(linspace(-2.0,0.0,n))
tau = collect(linspace(0.0,30.0,n))

tau = tau[2:end]

DR = 2.34
v0 = 13.19*2.0
mu = 31.624014716770823

bins_density = Float64.(logspace(1e-4,6.0,100))
bins_time = 100

Data = Dict()
prog = Progress(runs*n*n*length(Ps))
for a in alpha
    for b in tau
        for p in 1:length(Ps)
            #@info Ps[p]
            pdfs_model = zeros(runs,length(bins_density))
            Ts = zeros(runs,len,Ps[p],2)
            for r in 1:runs
                D = Sim(Ps[p],a,b,v0,mu,1.);
                for i in 1:size(Ts,2)
                    for j in 1:size(Ts,3)
                        # extract the trajectories
                        Ts[r,i,j,:] = D[(i-1)*Ps[p]+j,:]
                    end
                end
                if b == 0.0
                    # density is not recorded in simulation since the pre-factor is 0.0
                    d = zeros(size(Ts,2),size(Ts,3))
                    for i in 1:size(Ts,2)
                        d[i,:] = Density(Ts[r,i,:,1:2]./2.0)
                    end
                    d = flatten(d)
                else
                    # density data from CUDA code
                    d = readdlm("trajectories.txt",',')[:,4]
                end
                # extract second half of data
                h = Int(floor(size(d,1)/2.))
                kde = stats.gaussian_kde(d[h:end])
                pdfs_model[r,:] = kde(bins_density)
                next!(prog)
            end
            # save to dict
            Data["$(Ps[p])-$a-$b"] = pdfs_model
	    Data["Trajectory-$(Ps[p])-$a-$b"] = Ts
        end
    end
end
# write to this dir
JLD2.@save "alpha-tau-pd-15.jld" Data
