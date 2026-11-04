using GameTracing
using Test

include("games.jl")
include("fixtures.jl")
include("arms.jl")
include("gambit.jl")

"""
Include `file`. A file past `seconds` is stuck: a separate process SIGQUITs this one, printing every thread's
backtrace. The watchdog's shell dies when the file ends, so its timeout then quits nothing.
"""
function watched_include(file::AbstractString, seconds::Integer)::Nothing
    watchdog = run(`sh -c "timeout $seconds tail --pid=$(getpid()) -f /dev/null || kill -QUIT $(getpid())"`; wait = false)
    try
        include(file)
    finally
        kill(watchdog)
    end
    return nothing
end

const GAME = LinearGainPass(Float32[0 1; 1 0], Float32[0.5f0, -0.25f0])
const TARGET_PRECISION = 2.0f0
const WIDTH = 2

# Every `*_tests.jl` file runs, in name order, so none can be left out; the code rules run last. One outer testset
# keeps a failing file from aborting the rest: a failed top-level testset throws as it ends.
@testset "GameTracing" begin
    for file in sort(filter(endswith("_tests.jl"), readdir(@__DIR__; join = true)))
        # The slowest file takes about 2 minutes on CI; 10 leaves room for a slow runner.
        watched_include(file, 600)
    end
    include("code_rules.jl")
end
