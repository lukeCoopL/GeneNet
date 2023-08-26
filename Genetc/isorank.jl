import Pkg; Pkg.add("LinearMaps")
import Pkg; Pkg.add("DataStructures")
import Pkg; Pkg.add("LightGraphs")
import Pkg; Pkg.add("SparseArrays")
import Pkg; Pkg.add("LinearAlgebra")
import Pkg; Pkg.add("Random")
#import Pkg; Pkg.add("IsoRank")
#import Pkg; Pkg.clone("https://github.com/vvjn/IsoRank.jl")
#__precompile__()

#module IsoRank

using LinearMaps
using DataStructures
using LinearAlgebra
#export isorank, greedyalign

"""
    kronlm([T], A, B)
Kronecker product of `A` and `B`, stored as a linear operator (from
[LinearMaps.jl](https://github.com/Jutho/LinearMaps.jl.git)) so that
you don't have to create the actual matrix, i.e. `kronlm(A,B)*x == kron(A,B)*x`.
This is much faster than
directly creating the matrix: `O(|V|^2+|V||E|)` instead of `O(|V|^2 + |E|^2)` for each
step of the power iteration where `|V|` and `|E|` are the number of nodes and edges
in the graphs.
# Arguments
- `A,B` : linear operators with multiply and transpose operations
- `T` : element type of the resulting linear operator 
"""
function kronlm(::Type{T},A,B) where {T}
    f = (y,x) -> begin
        y[:] = vec(B * reshape(x, size(B,2), size(A,2)) * A')
        y end
    fc = (y,x) -> begin
        y[:] = vec(B' * reshape(x, size(B,1), size(A,1)) * A)
        y end
    m = size(A,1) * size(B,1)
    n = size(A,2) * size(B,2)
    LinearMap{T}(f, fc, m, n)
end
kronlm(A,B) = kronlm(promote_type(eltype(A),eltype(B)),A,B)

"""
    powermethod!(A, x, Ax=similar(x);
                 <keyword arguments>) -> radius, x, [log/history]
Performs power method in order to find the dominant eigenvector
of the linear operator A. Eigenvector is normalized w.r.t. L_1 norm.
Modifies initial eigenvector estimate x.
# Arguments
- `A` : linear operator
- `x` : initial estimate of the eigenvector, not necessarily normalized
- `Ax` : scratch space for the iteration process    
# Keyword arguments
- `maxiter` : maximum # of iterations
- `tol=eps(Float64) * size(A,2)` : error tolerance in L_1 norm
- `log=true,verbose=true` : logging and printing
"""    
function powermethod!(A, x, Ax=similar(x);
                      maxiter=15, tol=eps(Float64) * size(A,2),
                      log=true, verbose=true)
    T = eltype(A)
    x ./= norm(x,1)
    verbose && println("Running power method, maxiter = $maxiter, tol = $tol")
    history = Tuple{Int,T,T}[]
    iter = 0
    radius = zero(T)
    err = Inf
    while iter <= maxiter
        mul!(Ax, A, x)
        radius = norm(Ax,1)
        Ax ./= radius # want |x|_1 = 1
        err = norm(Ax-x,1)
        verbose && @show iter,err,radius
        log && push!(history,(iter,err,radius))
        copy!(x, Ax)
        err < tol && break
        iter += 1
    end
    isconverged = err < tol
    if log radius,x,history,isconverged else radius,x end
end

"""
    isorank(G1::SparseMatrixCSC, G2::SparseMatrixCSC,
            [alpha::Real=0.85, B::AbstractMatrix=ones(Float64,size(G1,1),size(G2,1))];
            <keyword arguments>) -> R [, res, L]
Creates the IsoRank matrix, which contains topological similarities of
all nodes across two networks (See Rohit Singh, Jinbo Xu, and Bonnie
Berger. (2008) Global alignment of multiple protein interaction
networks with application to functional orthology detection,
Proc. Natl. Acad. Sci. USA, 105:12763-12768.). That is, finds the
PageRank values of the modified adjacency matrix of the product graph
of G1 and G2.  `B`, containing prior node similarities, acts as the
personalization vector, allowing you to incorporate external
information. If you don't have node similarities, you can still create
a good IsoRank matrix by damping like PageRank does.
# Arguments
- `G1,G2` : two adjacency matrices of undirected networks
- `B` : matrix of node similarities between `G1` and `G2`, not necessarily normalized
- `alpha`: weight between edge and node conservation
# Keyword arguments
- `details=false` : If true, returns (R,res,L) where R is the IsoRank
    matrix of node similarities, res is the power method detailed
    results structure, L is the linear operator that the power method
    finds the eigenvector of; if false, returns R
- See [`powermethod!`](@ref) for other keyword arguments
"""
function isorank(G1::AbstractMatrix, G2::AbstractMatrix,
                 alpha::Real=0.85, B::AbstractMatrix=ones(Float64,size(G1,1),size(G2,1));
                 details=false, args...)
    A = kronlm(Float64,G2,G1)
    res = pagerank(A, alpha, vec(B); details=details, args...)
    if details
        R = reshape(res[1], size(G1,1), size(G2,1))
        R, res[2], res[3], A
    else
        R = reshape(res, size(G1,1), size(G2,1))
        R
    end
end

"""
     pagerank(A, alpha=0.85, p = fill(1/size(A,2),size(A,2)),
              x=copy(p), Ax=similar(x);
              <keyword args>) -> x [, res, L]
Creates PageRank vector.
# Arguments
- `A` : Adjacency matrix of the graph. A[u,v] = 1 if u -> v
- `alpha` : Damping.
- `p` : Initial probability vector, or personalization vector, not necessarily normalized.
    
# Keyword arguments
- `details=false` : If true, returns (x,res,L) where x is the PageRank
  vector, res is the power method detailed results structure, L is the linear
  operator that the power method finds the eigenvector of; if false,
  returns x.
- See [`powermethod!`](@ref) for other keyword arguments   
"""    

#removed *sum(x[D]) from between alpha and +1.0 
function pagerank(A, alpha=0.85, p = fill(1/size(A,2),size(A,2));
                  details=false, args...)
    S = 1.0 ./ (A * ones(Float64,size(A,2)))
    D = findall(isinf,S) # S[i] = Inf if node i has no outlinks
    println(D)
    println(S)
    for i in D
        S[i] = 0.0
        
    end
    println(S)
    p = p./norm(p,1)
    L = LinearMap{Float64}((y,x) -> begin
                           mul!(y, A, S .* x)
                           sump=0
                           for i in D
                                sump=sump+x[i]
                           end
                           y .= alpha .* y .+ (alpha*sump + 1.0 - alpha) .* p
                           y
                           end, size(A,1), size(A,2))
    x = copy(p)
    res = powermethod!(L, x; args...)
    if details x, res, L else x end
end

"""
    greedyalign(R::AbstractMatrix,seeds=Vector{Tuple{Int,Int}}();
                maxiter=size(R,1)-length(seeds))
Given matrix `R`, find a alignment using the greedy method.
    
# Arguments
- `R` : m x n matrix
- `seeds` : Initial seed node pairs
- `maxiter` : Stop after aligning `maxiter` node pairs
"""    
function greedyalign(R::AbstractMatrix,seeds=Vector{Tuple{Int,Int}}();
                     maxiter=size(R,1)-length(seeds))
    f = zeros(Int,size(R,1))
    n1 = size(R,1); n2 = size(R,2)
    kv = (((i,j),-R[i,j]) for i=1:n1, j=1:n2)
    Q = PriorityQueue{Tuple{Int,Int},eltype(R)}(kv)
    L1 = Set{Int}()
    L2 = Set{Int}() # nodes already aligned
    for k = 1:length(seeds)
        i,j = seeds[k]
        f[i] = j
        push!(L1,i)
        push!(L2,j)
        for ip = setdiff(1:n1,L1); dequeue!(Q,(ip,j)); end
        for jp = setdiff(1:n2,L2); dequeue!(Q,(i,jp)); end
    end
    println("Building priority queue")
    iter = 1
    while iter <= maxiter && !isempty(Q)
        i,j = dequeue!(Q)
        f[i] = j
        push!(L1,i)
        push!(L2,j)
        for ip = setdiff(1:n1,L1); dequeue!(Q,(ip,j)); end
        for jp = setdiff(1:n2,L2); dequeue!(Q,(i,jp)); end
        print("\rIteration $iter/$maxiter")
        iter += 1
    end
    println()
    f
end



using LightGraphs, Random
g1 = erdos_renyi(10,0.1)
g2 = erdos_renyi(10,0.1)

G1 = adjacency_matrix(g1)
G2 = adjacency_matrix(g2)

R = isorank(G1, G2, 0.85)

R ./= maximum(R)
truemap = 1:size(G2,1)
randmap = randperm(size(G2,1))
#println(sum(R[sub2ind(size(R),truemap,truemap)]))
#println(sum(R[sub2ind(size(R),truemap,randmap)]))
f = greedyalign(R)
println(hcat(1:length(f), f[1:length(f)]))