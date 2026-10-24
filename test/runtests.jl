using GameTracing
using LinearAlgebra: LAPACKException, SingularException
using Random: RandomDevice, Xoshiro
using Test

include("games.jl")

const GAME = LinearGainPass(Float32[0 1; 1 0], Float32[0.5, -0.25])
const TARGET_PRECISION = 2.0f0
const WIDTH = 2

include("trace_tests.jl")
include("fold_tests.jl")
include("property_tests.jl")
include("failure_tests.jl")
