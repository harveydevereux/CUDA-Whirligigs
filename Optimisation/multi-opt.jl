using PyCall
using LinearAlgebra
using BlackBoxOptim
using DelimitedFiles
using Test
using Statistics

using ArgParse
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--gamma"
            arg_type = Float64
            default = 0.25
        "--max-hours"
            arg_type = Float64
            default = 1.0
		"--n-repeats"
			arg_type = Int
			default = 1
    end
    return parse_args(s)
end

args = parse_commandline()
@info args

# use python for loss to make sure it is consistent
stats = pyimport("scipy.stats")
np = pyimport("numpy")

N=199
kde_domain = (0,6)
dx = 0.1
n_runs = args["n-repeats"]
geometry = "circle"
v0 = 26.0 # speed in RADII per second as mean of collision free speed in 200 beetle data

D200 = readdlm("density-200.csv",',')[:,1]
D100 = readdlm("density-100.csv",',')[:,1]
D50 = readdlm("density-50.csv",',')[:,1]

kde_200 = stats.gaussian_kde(D200)
kde_100 = stats.gaussian_kde(D100)
kde_50 = stats.gaussian_kde(D50)

B_kde = [kde_50,kde_100,kde_200]

function MSEKDE(kde_p,kde_q,domain=(0,1),dx=0.1)
    X = np.linspace(domain[1],domain[2],Int(np.floor((domain[2]-domain[1])/dx)))
    return mean((kde_p(X).-kde_q(X)).^2.0)
end

@test @show MSEKDE(kde_200,kde_200) == 0.0

function BeetleModellLossCircleAllVids(p)
    try
        L = [0.0]
        alpha,tau,mu = p
        for i in 1:n_runs
            for (j,n) in enumerate([100])
				seed = Int(floor(time()))
                cmd = `../CUDAABP --initial-packing-fraction 0.5 -N $n -T 10 -a 1 -b 1 -alpha $alpha -tau $tau -dt 0.00111111 -mu $mu -mur 1 -v $v0 -Dr 2.34 --random-seed $seed -silent 1`
				run(cmd)
                T = readdlm("trajectories.txt",',')
                h = Int(ceil(size(T,1)/2.0))
                D = T[h:end,4]
                kde_data = stats.gaussian_kde(D)
                L[j] += MSEKDE(kde_data,B_kde[j+1],kde_domain,dx)
			end
		end
        print(p,L ./ n_runs)
        return Tuple(L ./ n_runs)
	catch e
		println(e)
		if typeof(e) == InterruptException
			return exit()
		end
        return 1e9
	end
end


@show v,t, = @timed BeetleModellLossCircleAllVids([-1.,1.5,30.])
@info "loss: $v"
@info "runtime: $t seconds"
MaxT = args["max-hours"]*60^2
γ = args["gamma"]
tot = Int(floor(MaxT/t))
pop = Int(floor(γ*MaxT / t))

@info "Running for $(args["max-hours"]) hours, initial random population $pop / $tot runs"

res = bboptimize(BeetleModellLossCircleAllVids,Method=:borg_moea,
    	FitnessScheme=ParetoFitnessScheme{1}(is_minimizing=true),
		SearchRange=[(-2.0,0.0),(0.0,50.0),(10.0,60.0)],
		MaxTime=MaxT,
		PopulationSize=pop)
