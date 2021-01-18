using DelimitedFiles
using Test
include("../Analysis/AlphaShapes/AlphaShapes.jl")
X = rand(1000,2).*100;
writedlm("./delaunay-test.txt",X,',')

function Density(X)
    tess,inds = AlphaShapes.GetDelaunayTriangulation(X,true);
	d = zeros(size(X,1))
	for i in 1:size(X,1)
    	d[i] = AlphaShapes.WeightedDTFELocalDensity(i,tess,inds)
	end
	return d
end

D_julia = Density(X)
run(`g++ -std=c++11 -O3 delaunator-test.cpp -o test`)
run(`./test delaunay-test.txt`)
D_cpp = readdlm("out.txt",'\n')[:,1]
D_julia.-D_cpp#

@show @test sum(abs.(D_julia .- D_cpp))/length(D_julia) < 1e-6

run(`rm test out.txt delaunay-test.txt`)
