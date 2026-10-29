# EXPERIMENT: trace the stall arms named in ARGS, in order, in one fresh process; not for landing.
# "seedN" seeds the global RNG. Each arm prints its steps and a fingerprint of the global RNG before it.
using GameTracing
using Test
using Random: Random

include("games.jl")
include("fixtures.jl")
include("arms.jl")
include("float32_stall_experiment.jl")

function order_report(arms::AbstractVector{<:AbstractString})::Nothing
    profile_status = LogitNash.solve(map(float, UNFOLDED_PITCHFORK_GAME); stop_lambda = float(4), stop_eps = zero(float(4)))
    precision = Float32(last(profile_status).lambda)
    for arm in arms
        if startswith(arm, "seed")
            Random.seed!(parse(Int, arm[5:end]))
            continue
        end
        fingerprint = rand(copy(Random.default_rng()), UInt32)
        E = ifelse(arm == "E", typeof(float(1)), Float32)
        settings = arm_settings(ARMS.fisher_log_precision_reach, E(1) / 100)
        result = trace_endpoint(stall_game(arm, E(precision)), length(UNFOLDED_PITCHFORK_GAME), settings)
        if is_error(result)
            outcome = string(unwrap_error(result).cause, " steps ", unwrap_error(result).context.steps)
        else
            outcome = string("steps ", unwrap(result).steps)
        end
        println("ORDER ", join(arms, ","), " arm ", arm, " ", outcome, " rng ", fingerprint)
    end
    return nothing
end

order_report(ARGS)
