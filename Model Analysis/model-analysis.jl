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
include("../Analysis/AlphaShapes/AlphaShapes.jl")
include("../Analysis/analysis-utils.jl")

# using CSV rather than Julia specific JLD format
X50 = readdlm("../Data/50-X.csv",',')
X50 = zeros(size(X50)...,3)
X50[:,:,1] = readdlm("../Data/50-X.csv",',')
X50[:,:,2] = readdlm("../Data/50-Y.csv",',')
X50[:,:,3] = readdlm("../Data/50-theta.csv",',')

X100 = readdlm("../Data/100-X.csv",',')
X100 = zeros(size(X100)...,3)
X100[:,:,1] = readdlm("../Data/100-X.csv",',')
X100[:,:,2] = readdlm("../Data/100-Y.csv",',')
X100[:,:,3] = readdlm("../Data/100-theta.csv",',')

X200 = readdlm("../Data/200-X.csv",',')
X200 = zeros(size(X200)...,3)
X200[:,:,1] = readdlm("../Data/200-X.csv",',')
X200[:,:,2] = readdlm("../Data/200-Y.csv",',')
X200[:,:,3] = readdlm("../Data/200-theta.csv",',');

JLD2.@load "../Data/alpha-tau-pd-15.jld"
# data is accesible from the global variable "Data" like
# Data["100--0.14285714285714285-0.0"]
# 3×100 Array{Float64,2}:
#  3.47599  3.47702  3.47817  3.47946  3.4809   …  0.0  0.0  0.0  0.0  0.0  0.0
#  3.91475  3.91589  3.91717  3.9186   3.92019     0.0  0.0  0.0  0.0  0.0  0.0
#  3.94935  3.9507   3.95221  3.9539   3.95578     0.0  0.0  0.0  0.0  0.0  0.0
# where each row is the density distribution of one run with N=100, alpha = -0.14285714285714285, and tau = 0.0

runs = 3
Ps = [50,100,200]
n = 15
# alpha and tau values used for the data
alpha = collect(linspace(-2.0,0.0,n))
tau = collect(linspace(0.0,30.0,n))

bins_density = Float64.(logspace(1e-4,6.0,100))
bins_time = 100

BX = bins_density
B_pdfs = [TimeBinDensity(BX,100,X50,length_scale=beetle_length),
          TimeBinDensity(BX,100,X100[500:end,:,:],length_scale=beetle_length),
          TimeBinDensity(BX,100,X200,length_scale=beetle_length)]

@info "Computing Phase Diagrams"
@info "Looping over Alpha-Tau"
prog = Progress((n)*(n)*length(Ps))
for p in 1:length(Ps)
  plot(layout=(length(alpha)+1)*(length(tau)+1),size=(10000,10000))
  plot!(xaxis=nothing,yaxis=nothing,bordercolor="white",subplot=1)
  # tau axis label
  for j in 1:n
      annotate!(0.5,0.5,"$(round(tau[j],digits=3))",xaxis=nothing,yaxis=nothing,bordercolor="white",100,subplot=j+1)
  end
  # alpha axis label
  for j in 1:n
      annotate!(0.5,0.5,"$(round(reverse(alpha)[j],digits=3))",xaxis=nothing,yaxis=nothing,bordercolor="white",100,subplot=j*(n+1)+1)
  end
  i = n+2
  for a in reverse(alpha)
      for b in tau
          pdfs = Data["$(Ps[p])-$a-$b"]
          e = std(pdfs,dims=1)[1,:]
          pdfs = mean(pdfs,dims=1)[1,:]
          plot!(BX,pdfs,ribbon=e,label="",c="red",subplot=i+1,title="")
          plot!(BX,mean(B_pdfs[p],dims=1)[1,:],ribbon=std(B_pdfs[p],dims=1)[1,:],label="",c="blue",subplot=i+1,fillalpha=0.1)
          i += 1
          next!(prog)
          ylims!(0.0,Inf) # min 0.0, but max found as needed
          if p == 1
              xlims!(0,2.0)
          else
              xlims!(0,3.5)
          end
      end
      i += 1
  end
  savefig("alpha-tau-pd-$(Ps[p]).png")
end
