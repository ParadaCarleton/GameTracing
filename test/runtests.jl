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
    # Arm order ran on 324c61a (run 40819775696): B2 took 35 steps in every order. Now: the CPU's kernels and codegen.
    println("MACHINE cpu_name=", Sys.CPU_NAME, " model=", first(Sys.cpu_info()).model)
    script = joinpath(@__DIR__, "float32_stall_order.jl")
    for coretype in ("", "Haswell", "Zen", "SkylakeX", "Sandybridge", "Prescott")
        println("OPENBLAS_CORETYPE=", coretype)
        run(ignorestatus(addenv(`$(Base.julia_cmd()) --project=$(Base.active_project()) $script B2 B3 B5`, "OPENBLAS_CORETYPE" => coretype)))
    end
    for target in ("generic", "haswell", "x86-64-v3", "x86-64-v4")
        println("julia -C ", target)
        run(ignorestatus(`$(Base.julia_cmd()) -C $target --project=$(Base.active_project()) $script B2 B3 B5`))
    end
    @test true
end
