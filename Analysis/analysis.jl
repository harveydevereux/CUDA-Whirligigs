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

image_type = "png"

using ArgParse
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--pdf"
            action = :store_true
        "--svg"
            action = :store_true
    end
    return parse_args(s)
end

args = parse_commandline()
@info args

if args["pdf"]
    image_type="pdf"
elseif args["svg"]
    image_type="svg"
elseif args["pdf"] && args["svg"]
    @info "Multiple filetypes, defaulting to png"
end


@info "Saving plots as .$(image_type)"

beetle_length=13.0
beetle_width=7.0
dt = 1.0/30.0

include("AlphaShapes/AlphaShapes.jl")
include("analysis-utils.jl")

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

# collect speed and density
@info "collecting speed and density distributions"
D = []
V = []
@showprogress for X in [X50,X100,X200]
    d = zeros(size(X,1),size(X,2))
    v = zeros(size(X,1)-1,size(X,2))
    for i in 1:size(X,1)
        d[i,:] = Density(X[i,:,1:2] ./ beetle_length)
        if 1 < i < size(X,1)-1
            for j in 1:size(X,2)
                # central difference about i
                v[i,j] = norm(X[i+1,j,1:2] .- X[i-1,j,1:2])./(2.0*beetle_length*dt)
            end
        end
    end
    push!(D,d)
    push!(V,v)
end

names = ["50","100","200"]
linestyles = [:dash,:solid,:dot]
plot(layout=3,dpi=600,size=(900,900))
Xs = [X50,X100,X200]
for i in 1:size(Xs,1)
    X = Xs[i]
    bx = Float64.(logspace(1.0e-4,1.0,100))
    # bin across time and find avg and std for density
    # pdf across time bins
    pdfs = TimeBinDensity(bx,100,X,length_scale=beetle_length)
    pdf = mean(pdfs,dims=1)[1,:]
    pdf_e = std(pdfs,dims=1)[1,:]
    plot!(bx,pdf,ribbon=pdf_e,label="N=$(names[i])",linestyle=linestyles[i],linewidth=3,subplot=i)
    # effectively 0.0 at this point
    xlims!(0.0,1.0)
    #ylims!(0.0,5.0)
end
plot!()
ylabel!("PDF")
xlabel!("Local Density (square body lengths)")
savefig("density-distributions-time-errors.$(image_type)")

names = ["50","100","200"]
linestyles = [:dash,:solid,:dot]
plot(dpi=600)
for i in 1:size(D,1)
    d = D[i][isnan.(D[i]).==false];
    pdf = kde(d)
    plot!(pdf.x,pdf.density,label="N=$(names[i])",linestyle=linestyles[i],linewidth=3)
end
plot!()
# effectively 0.0 at this point
xlims!(0,1.0)
ylabel!("Kernel Density Estimate of PDF")
xlabel!("Local Density (square body lengths)")
savefig("density-distributions.$(image_type)")

names = ["50","100","200"]
linestyles = [:dash,:solid,:dot]
plot(layout=3,dpi=150,size=(600,600))
for i in 1:size(V,1)
    v = V[i][isnan.(V[i]).==false]
    speed_bins=linspace(0.0,60.0,100)
    pdfs = TimeBinDistribution(speed_bins,v;bins=100)
    pdf = mean(pdfs,dims=1)[1,:]
    pdf_e = std(pdfs,dims=1)[1,:]
    plot!(speed_bins,pdf,ribbon=pdf_e,label="N=$(names[i])",linestyle=linestyles[i],linewidth=3,subplot=i)
    ylims!(0.0,maximum(pdf)+0.5,subplot=i)
end
plot!()
ylabel!("PDF")
xlabel!("Speed (body lengths per second)")
savefig("speed-distributions.$(image_type)")


# speed and density relationship data
@info "Computing speed and density relationship"

DV50 = []
DV100 = []
DV200 = []

DV = [DV50,DV100,DV200]
@showprogress for (i,X) in enumerate([X50,X100,X200])
    for t in 1:size(X,1)-1
        for b in 1:size(X,2)
            push!(DV[i],[D[i][t,b],V[i][t,b],X[t,b,3]])
        end
    end
end

@. model(x,p) = p[1].*x.^(-p[2]).+p[3]
n = ["50","100","200"]
Data = []
fits = []
for (j,DV) in enumerate([DV50,DV100,DV200])
    d = []
    v = []
    o = []
    for i in 1:length(DV)
        push!(d,DV[i][1])
        push!(v,DV[i][2])
        push!(o,DV[i][3])
    end

    ind = isinf.(d).==false
    d = d[ind]
    v = v[ind]
    o = o[ind]

    ind = (d) .< Inf
    # scale both back to pixels for plot function
    d = d[ind] ./ beetle_length^2.0
    v = v[ind] .* (2.0*dt*beetle_length)
    o = o[ind]

    #println(maximum(d))

    # generate a binned plot with errorbars
    # 1.0/(2dt) because of the central difference!
    local D,V,SE=DVBinPlot(D=d,V=v,n_bins=25,beetlescale=true,beetle_length=beetle_length,xscale=:log10,FPS=1.0/(2.0*dt),errors=true)
    inds = isnan.(D) .== false
    D = D[inds]
    V = V[inds]
    inds = isnan.(V) .== false
    D = D[inds]
    V = V[inds]

    inds = isnan.(o) .== false
    d = d[inds]
    o = o[inds]
    d_,o_,seo_ = DVBinPlot(D=d,V=o,n_bins=25,beetlescale=false,errors=true)
    push!(Data,(D,V,SE,d_,o_,seo_))
    fit = curve_fit(model, D, V, [1.,1.,1.])
    @info "power law fit converged: ", fit.converged, " N = $(n[j])"
    push!(fits,fit.param)
end

replace!(Data[2][3],NaN=>0.0)
plot(dpi=600)
labels=[L"N=50",L"N=100",L"N=200"]
colours = ["red","green","blue"]
shapes = [:square, :diamond, :circle]
for i in 1:length(Data)
    scatter!(Data[i][1],Data[i][2],yerr=Data[i][3],label=labels[i],c=colours[i],mark=shapes[i])
end
plot!(xaxis=:log10,yaxis=:log10)
xlabel!(L"\rho")
ylabel!(L"v(\rho)")

ρ = logspace(1e-2,1,100)
plot!(ρ,10*ρ.^-0.4,label=L"\propto\rho^{-0.4}",c="black")
savefig("speed-density-relationship.$(image_type)")

# Diffusion
@info "Diffusion statistics"
T = copy(X200)

# unwravel phase
for i in 1:size(T,2)
    T[:,i,3] = unwrap(T[:,i,3])
end

MSAD,t = msdθ(T,size(T,1),800)
MSAD = MSAD[:,1]
@. model(t,p) = p[1]*t
ind = findfirst(x->x.>2.5,t*dt)
fit_ = curve_fit(model,t[ind:end].*dt,MSAD[ind:end],[1.0])
fit_.converged

k = 15

x = t[1:k:end]*dt
y = MSAD[1:k:end]
xt = t[ind:end]*dt
yt = model(t[ind:end]*dt,fit_.param)

xt = xt[1:k:end]
yt = yt[1:k:end]

scatter(x,y,label="",legend=:topleft,title="")
plot!(xt,yt,label="Fit to 2.0*Dr*t, Dr = $(round(fit_.param[1]/2.0,digits=3))",linewidth=2)
xlabel!("Seconds")
ylabel!("Mean Square Angular Displacement (rad^2 s)")

T = copy(X200)
Dr = 2.341
taur = 1.0 / Dr

MSAD,t = MSD(T.*(5e-3/beetle_length),size(T,1),30*1)
MSAD = MSAD[:,1];

@. model(t,p) = (4.0*p[1]+(2.0*p[2]^2.0)*taur)*t + (2.0*p[2]^2.0)*(taur^2.0)*(exp(-t/taur)-1)
ind = 1
fit_ = curve_fit(model,t[ind:end].*dt,MSAD[ind:end],[1.0,1.0],lower=[0.0,5e-3], upper=[Inf,Inf])
fit_.converged

scatter(t*dt,MSAD,label="",legend=:topleft,title="",dpi=600)
plot!(t[ind:end]*dt,model(t[ind:end]*dt,fit_.param),label="Dt = $(round(fit_.param[1],digits=5)), v0 = $(round(fit_.param[2],digits=5))")
xlabel!("Seconds")
ylabel!("Mean Square Displacement (m^2 s)")
savefig("translational-diffusion.$(image_type)")

@info "Example of delay between velocity and orientation"
b = 3
t_max = 20
K = 5
X = X50
plot(aspect_ratio=:equal,grid=false,dpi=600)
plot!(xaxis=nothing,yaxis=nothing)
for t in 2:t_max
    H = VelocityHeading(X,t,b)
    o = [cos(X[t,b,3]),sin(X[t,b,3])]
    Arrow(H,X[t,b,1:2],K,"black")
    Arrow(o,X[t,b,1:2],K,"red")
    E = Ellipse(1000,a=13.0/2.0,b=7.0/2.0,θ=X[t,b,3],center=X[t,b,1:2])
    s = Shape(E[:,1],E[:,2])
    plot!(s,aspect_ratio=:equal,c="blue",alpha=0.25,label="")
end
plot!([],[],label="Body Orientation",c="red")
plot!([],[],label="Velocity Direction",c="Black")
plot!([],[],label="Template Indicative of Beetle Outline",c="blue",alpha=0.25)
savefig("delay-example.$(image_type)")

### CollisonFreeAnalysis

@info "Collision Free Statistics"

data = [X200]
C = Collisions(beetle_length,data) # collision filtered
T,L,O = CollisionFreeTrajectories(C[1],X200)

histogram(L,label="",dpi=600)
savefig("collision-free-trajectory-length-distribution.$(image_type)")

@info "Gathering orientation Velocity Correlation"

C,t = NVCorr(O,1.0/30.0,4,2);
c = length.(C)
e = std.(C)
Cm = mean.(C)
t = t[2:end]
Cm = Cm[2:end]

@. model(t,p) = p[1].*exp.(-0.5.*((t.-p[2])^2.0)./p[3]^2.0)
fit_ = curve_fit(model,t,Cm,[1.0,1.0,1.0])
@info  fit_.converged,fit_.param
@info "Gaussian fit peak location ", fit_.param[2]
@info "MSE ", mean((Cm.-model(t,fit_.param)).^2.0)
tt = collect(minimum(t):0.001:maximum(t))

p1 = scatter(t,Cm,yerr=e./sqrt.(c),label="")
plot!(tt,model(tt,fit_.param),label="Gaussian Fit",linewidth=3,xtickfontsize=18,ytickfontsize=18)
savefig("velocity-orientation-correlation-short-times.$(image_type)")

C,t = NVCorr(O,1.0/30.0,30,2);
c = length.(C)
e = std.(C)
Cm = mean.(C)
t = t[2:end]
Cm = Cm[2:end]

p2 = scatter(t,Cm,yerr=e./sqrt.(c),label="")
plot!(tt,model(tt,fit_.param),label="Gaussian Fit",linewidth=3)
savefig("velocity-orientation-correlation.$(image_type)")

plot(collect(2:size(n,1)+1).*1.0/30.0,n,label="")
xlabel!("Minimum Collision Free Trajectory Length (seconds)")
ylabel!("Number of Trajectories")
savefig("collision-free-trajectory-count.$(image_type)")


data = [X50,X100,X200]
C = Collisions(beetle_length,data)
for j in 1:3
    local t,l,o = CollisionFreeTrajectories(C[j],data[j])
    local S = []
    @showprogress for i in 1:length(o)
        d = o[i]
        if length(d) >= 3
            for t in 2:size(d,1)-1
                v = d[t+1][1:2] .- d[t-1][1:2]
                if isnan(norm(v)) == false
                    push!(S,norm(v)/(2*beetle_length*dt))
                end
            end
        end
    end
    @info "N = $(names[j]): Mean Collision Free Speed", mean(S), " body lengths per second"
end
