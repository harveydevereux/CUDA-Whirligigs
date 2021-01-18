- Slow analysis.jl (Warning: implicit `dims=2`)
  - In Julia 1.4, Distances v0.9.2 suffers a large performance hit if dims=2 is not specified, compare the following snippets
    ```bash 
    $ julia-1.4 -e "using Distances; pairwise(Euclidean(),randn(50,2)'); @show (@timed pairwise(Euclidean(),randn(50,2)'))[2]"
    ┌ Warning: implicit `dims=2` argument now has to be passed explicitly to specify that distances between columns should be computed
    │   caller = ip:0x0
    └ @ Core :-1
    (#= none:1 =# @timed(pairwise(Euclidean(), (randn(50, 2))')))[2] = 0.013124105
    ```
    
    ```bash
    $ julia-1.5 -e "using Distances; pairwise(Euclidean(),randn(50,2)'); @show (@timed pairwise(Euclidean(),randn(50,2)'))[2]"
    (#= none:1 =# @timed(pairwise(Euclidean(), (randn(50, 2))')))[2] = 3.0775e-5
    ```
   - Best fix is to use Julia 1.5
