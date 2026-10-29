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
    # A, B and E ran in run 40814179626, C and D in run 40814692206.
    # B1 to B5 ran in run 40815333022, the A and B controls on a3bde83.
    # Arm order: does B2's step count depend on the arms traced before it in the process?
    for arms in (["B2"], ["B2"], ["A", "B4", "E", "B2"], ["E", "B1", "B2"], ["E", "A", "B", "B1", "B2"],["seed1", "B2"], ["seed2", "B2"])
        run(`$(Base.julia_cmd()) --project=$(Base.active_project()) $(joinpath(@__DIR__, "float32_stall_order.jl")) $arms`)
    end
    @test true
end
