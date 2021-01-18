function AngleToX360(v)
    θ = angle(Complex(v...))
    if θ < 0
        θ = 2*π+θ
    end
    return θ
end

function Angle(u,v)
    α = (u⋅v)/(norm(u)*norm(v))
    if abs(α) > 1
        α=sign(α)*1
    end
    return acos(α)
end

function unwrap(v, inplace=false)
  # currently assuming an array
  unwrapped = inplace ? v : copy(v)
  for i in 2:length(v)
    while unwrapped[i] - unwrapped[i-1] >= pi/2
      unwrapped[i] -= pi
    end
    while unwrapped[i] - unwrapped[i-1] <= -pi/2
      unwrapped[i] += pi
    end
  end
  return unwrapped
end

function Ellipse(n;a=1,b=1,max=2π,θ=0,center=[0.0,0.0])
    x = [a*cos(t)*cos(θ) - b*sin(t)*sin(θ) for t in 0:(2π/n):2π] .+ center[1]
    y = [a*cos(t)*sin(θ) + b*sin(t)*cos(θ) for t in 0:(2π/n):2π] .+ center[2]
    return cat(x,y,dims=2)
end

beetle_length = 13.0
beetle_width = 7.0
dt = 1.0/30.0


function Density(X)
    tess,inds = AlphaShapes.GetDelaunayTriangulation(X,true);
	d = zeros(size(X,1))
	for i in 1:size(X,1)
    	d[i] = AlphaShapes.WeightedDTFELocalDensity(i,tess,inds)
	end
	return d
end

flatten(x) = reshape(x,prod(size(x)))

function logspace(start,stop,n=10)
    s = log10(start)
    S = log10(stop)
    x = []
    for i in linspace(s,S,n)
        push!(x,10^i)
    end
    return x
end

function linspace(l,u,n=10)
    return collect(range(l,stop=u,length=n))
end

function DVBinPlot(;D,V,n_bins=10,beetlescale=true,beetle_length=14.0,xscale=:log10,FPS=30.0,errors=true)
    IFT = 1/FPS
    bin_values = logspace(minimum(D),maximum(D),n_bins)

    Bin_V_SE = [[] for i in 1:n_bins]
    Bin_V = zeros(n_bins)
    number = zeros(n_bins)

    for b in 1:(n_bins-1)
        ind1 = findall(x->x.>=bin_values[b],D)
        v = V[ind1]
        ind = findall(x->x.<bin_values[b+1],D[ind1])
        if size(ind,1) > 0
            Bin_V[b] += sum(v[ind])
            append!(Bin_V_SE[b],v[ind])
            number[b] += size(ind,1)
        end
    end

    if beetlescale
        SE = zeros(n_bins)
        for i in 1:n_bins
            if size(Bin_V_SE[i],1) > 0
                SE[i] = std(Float64.(Bin_V_SE[i]) ./ (beetle_length * IFT)) / sqrt(number[i])
            end
        end
        Bin_V = Bin_V[2:end-1] ./ ((number[2:end-1] .* beetle_length) * IFT)
        bin_values = bin_values[2:end-1].*beetle_length^2.0
        if errors
            scatter(bin_values,Bin_V,xscale=xscale,yerr=SE,label="")
        else
            plot(bin_values,Bin_V,xscale=xscale,label="")
        end
    else
        SE = zeros(n_bins)
        for i in 1:n_bins
            if size(Bin_V_SE[i],1) > 0
                SE[i] = std(Float64.(Bin_V_SE[i])) / sqrt(number[i])
            end
        end
        if errors
            plot(bin_values,Bin_V,xscale=xscale,yerr=SE,label="")
        else
            plot(bin_values,Bin_V,xscale=xscale,label="")
        end
    end
    println(number)
    return bin_values,Bin_V,SE[2:end-1]
end

function MSD(X,t=size(X,1),lags=120)
    l = 0:lags
    D = zeros(size(l,1))
    C = zeros(size(l,1),size(X,1))
    @showprogress for i in 1:t
            for τ in l
                if i+τ > size(X,1)
                    continue
                end
                for j in 1:size(X,2)
                    p = norm(X[i+τ,j,1:2] - X[i,j,1:2])^2
                    if isnan(p) == false
                        D[τ+1] += p
                        C[τ+1,i] += 1
                    end
                end
            end
        end
    return D ./ sum(C,dims=2),collect(l)
end

function msdθ(X,t=size(X,1),lags=30)
    l = 0:lags
    D = zeros(size(l,1))
    A = X[:,:,3]
    C = zeros(size(l,1),t)
    @showprogress for i in 1:t
            for τ in l
                if i+τ > size(A,1)
                    continue
                end
                for j in 1:size(X,2)
                    p = (A[i+τ,j] - A[i,j])^2
                    if isnan(p) == false
                        D[τ+1] += p
                        C[τ+1,i] += 1
                    end
                end
            end
        end
    return D ./ sum(C,dims=2),collect(l)
end

function VelocityHeading(X,t,b)
    h = X[t+1,b,1:2].-X[t,b,1:2]
    return h./norm(h)
end

function Arrow(d,pos,K=1,c="red",linewidth=1)
    v = ones(2).*d*K + pos
    plot!([pos[1],v[1]],[pos[2],v[2]],color=c,label="",linewidth=linewidth,arrow=arrow())
end

function TimeBinDensity(BX,bins,T;length_scale=2.0)
    times = Int.(floor.(range(1,stop=size(T,1),length=bins)))
    collect(times)
    pdfs = zeros(bins-1,length(BX))
    for t in 1:size(times,1)-1
        start = times[t]
        stop = times[t+1]
        Y = T[start:stop,:,1:2]

        D = zeros(size(Y,1),size(Y,2))
        for i in 1:size(Y,1)
            D[i,:] = Density(Y[i,:,1:2]./length_scale)
        end
        D = replace(D,NaN => 0.0)
        x = flatten(D)
        x = x[x.>0]
        x = x[x.<Inf]

        h = fit(Histogram,x,BX)
        h = normalize(h, mode=:density)
        h.weights = h.weights./size(x,1)
        push!(h.weights,0.0)
        pdfs[t,:] = h.weights
    end
    return pdfs
end

function TimeBinDistribution(x,data;bins=10)
    times = Int.(floor.(range(1,stop=size(data,1),length=bins)))
    collect(times)
    pdfs = zeros(bins-1,length(x))
    for t in 1:size(times,1)-1
        start = times[t]
        stop = times[t+1]
        Y = []
        for i in start:stop
            for j in 1:size(data[i],1)
                push!(Y,data[i][j])
            end
        end
        y = replace!(Y,NaN => 0.0)
        y = y[y.>0]
        y = y[y.<Inf]

		h = fit(Histogram,y,x)
		h = normalize(h, mode=:density)
		h.weights = h.weights./size(x,1)
		push!(h.weights,0.0)
		pdfs[t,:] = h.weights
    end
    return pdfs
end

## Collision Filtering

using Distances

function Collisions(R=13.0,data=[X200])
    R = 13.0
    Data = []
    Cs = []
    for X in data
        C = Bool.(zeros(size(X,1),size(X,2),size(X,2)))
        for t in 1:size(X,1)
            d = pairwise(Euclidean(),X[t,:,1:2]').<=R
            for b in 1:size(X,2)
                for j in findall(x->x.==1,d[b,:])
                    C[t,b,j] = 1
                end
                C[t,b,b] = 0
            end
        end

        push!(Cs,C)

        for b in 1:size(C,2)
            flag = sum(C[1,b,:]) > 0
            d = []
            for t in 2:size(C,1)
                if sum(C[t,b,:]) > 0
                    push!(d,[t,X[t,b,:]...,findall(x->x.==1,C[t,b,:])...])
                    flag = 0
                end
                if sum(C[t,b,:]) == 0
                    if flag == 0
                        push!(Data,d)
                        d = []
                    end
                    flag = 1
                end
            end
            push!(Data,d)
        end
    end
    @info length(Data)
    return Cs
end

function CollisionFreeTrajectories(C,X)
    T = []
    d = zeros(size(X,1),size(X,2))
    for i in 1:size(X,1)
        d[i,:] = Density(X[i,:,1:2]./13.0)
    end
    @showprogress for i in 1:size(C,2)
        c = sum(C[:,i,:],dims=2)[:,1]
        flag = c[1]
        for j in 1:size(c,1)
            tau = []
            offset = 0
            while length(findall(x->x.==0,c)) > 0
                s = findfirst(x->x.==0,c)
                e = findfirst(x->x.==1,c)
                if e == nothing
                    e = length(c)
                end
                for k in s:1:e
                    push!(tau,([X[offset+k,i,:]...,d[offset+k,i]],offset+k))
                end
                offset += e
                c = c[(e+1):end]
            end
            if length(tau) > 0
                push!(T,tau)
            end
        end
    end

    # gets length count distribution
    L = []
    O = []
    for i in 1:size(T,1)
        t = T[i]
        if length(t) > 1
            o = [t[1][1]]
            k = t[1][2]
            l = 0
            for i in 2:size(t,1)
                if t[i][2] - k == 1
                    l += 1
                    k += 1
                    push!(o,t[i][1])
                else
                    k = t[i][2]
                    push!(L,l)
                    l = 0
                    o = [t[i][1]]
                    push!(O,o)
                end
            end
        end
    end
    return T,L,O
end

function MSDθ_CF(data,dt=1.0/30.0,lag_max=30)
    C = [[] for i in 1:(lag_max+1)]
    @showprogress for i in 1:length(data)
        d = data[i]
        if length(d) >= 2
            for t in 1:size(d,1)
                for (i,tau) in enumerate(0:lag_max)
                    if 1 < t+tau < size(d,1)
                        c = (d[t+tau][3] - d[t][3])^2.0
                        if isnan(c) == false
                            push!(C[i],c)
                        end
                    end
                end
            end
        end
    end
    return C,collect(0:lag_max).*dt
end

function MSD_CF(data,dt=1.0/30.0,lag_max=30)
    C = [[] for i in 1:(lag_max+1)]
    @showprogress for i in 1:length(data)
        d = data[i]
        if length(d) >= 2
            for t in 1:size(d,1)
                for (i,tau) in enumerate(0:lag_max)
                    if 1 < t+tau < size(d,1)
                        c = (norm(d[t+tau][1:2] .- d[t][1:2]))^2.0
                        if isnan(c) == false
                            push!(C[i],c)
                        end
                    end
                end
            end
        end
    end
    return C,collect(0:lag_max).*dt
end

function NVCorr(data,dt=1.0/30.0,lag_max=30,min_length=2,info=false)
    C = [[] for i in 1:(2*lag_max+1)]
    n = 0
    @showprogress for i in 1:length(data)
        d = data[i]
        if length(d) >= min_length
            n += 1
			# central difference
            for t in 1:size(d,1)
                for (i,tau) in enumerate(-lag_max:1:lag_max)
                    if 1 < t+tau < size(d,1)-1
						# central difference about t+tau
                        h = d[t+tau+1][1:2] .- d[t+tau-1][1:2]
                        h = h ./ norm(h)
                        o = [cos(d[t][3]),sin(d[t][3])]
                        c = h⋅o
                        if isnan(c) == false
                            push!(C[i],c)
                        end
                    end
                end
            end
        end
    end
    if info
        @info n
    end
    return C,collect(-lag_max:1:lag_max).*dt
end
