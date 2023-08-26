Pkg.clone("https://github.com/vvjn/IsoRank.jl")
using IsoRank, LightGraphs

g1 = erdos_renyi(200,0.1)
g2 = g1

G1 = adjacency_matrix(g1)
G2 = adjacency_matrix(g2)

R = isorank(G1, G2, 0.85)

R ./= maximum(R)
truemap = 1:size(G2,1)
randmap = randperm(size(G2,1))
println(sum(R[sub2ind(size(R),truemap,truemap)]))
println(sum(R[sub2ind(size(R),truemap,randmap)]))