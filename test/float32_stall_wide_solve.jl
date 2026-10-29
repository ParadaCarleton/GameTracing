# EXPERIMENT: arms C and D, whose Float32 traces factor the bordered system in Float64 (`wide_bls`); not for landing.
# A process of its own, since the method below replaces every Float32 trace's bordered solver.
using GameTracing
using Test

include("games.jl")
include("fixtures.jl")
include("arms.jl")
include("float32_stall_experiment.jl")

function GameTracing.matrix_bls(width::Integer, ::Type{E})::WideBLS where {E <: Float32}
    return wide_bls(width, E)
end

stall_experiment(("C", "D"))
