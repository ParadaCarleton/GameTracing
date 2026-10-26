using GameTracing
using Test

include("games.jl")

const GAME = LinearGainPass(Float32[0 1; 1 0], Float32[0.5f0, -0.25f0])
const TARGET_PRECISION = 2.0f0
const WIDTH = 2

# Every `*_tests.jl` file runs, in name order, so none can be left out; the code rules run last.
for file in sort(filter(endswith("_tests.jl"), readdir(@__DIR__; join = true)))
    include(file)
end
include("code_rules.jl")
