using GameTracing
using Test

include("games.jl")
include("fixtures.jl")
include("arms.jl")
include("gambit.jl")

const GAME = LinearGainPass(Float32[0 1; 1 0], Float32[0.5f0, -0.25f0])
const TARGET_PRECISION = 2.0f0
const WIDTH = 2

# Every `*_tests.jl` file runs, in name order, so none can be left out; the code rules run last. One outer testset
# keeps a failing file from aborting the rest: a failed top-level testset throws as it ends.
# EXPERIMENT: this branch runs only the Float32 stall arms; C and D run in their own process (float32_stall_wide_solve.jl).
@testset "GameTracing" begin
    include("float32_stall_experiment.jl")
    stall_experiment(("A", "B", "E"))
    run(`$(Base.julia_cmd()) --project=$(Base.active_project()) $(joinpath(@__DIR__, "float32_stall_wide_solve.jl"))`)
    @test true
end
