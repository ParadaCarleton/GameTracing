using GameTracing
using Test

include("games.jl")
include("fixtures.jl")
include("arms.jl")

const GAME = LinearGainPass(Float32[0 1; 1 0], Float32[0.5f0, -0.25f0])
const TARGET_PRECISION = 2.0f0
const WIDTH = 2

include("trace_tests.jl")
include("fold_tests.jl")
include("property_tests.jl")
include("bordered_tests.jl")
include("failure_tests.jl")
include("logitnash_tests.jl")
include("geometry_tests.jl")
include("result_tests.jl")
include("projection_tests.jl")
include("swap_fixture_tests.jl")
include("code_rules.jl")
