# The trace-geometry arms in payoff gaps: step distance (summed or Fisher) × acceptance, each at three state weights.
# KL + angle accepts by KL and chord angle with the reach and first-step distance tests off; reach keeps the shipped ones.
const KL_ANGLE = CorrectorSettings{Float32}(reach = nothing, max_kl = 0.01f0, max_angle = 0.5f0)
const NO_DISTANCE_LIMIT = StepControl{Float32}(distance_limit = nothing)
const ACCEPTANCES = (
    kl_angle = (; corrector = KL_ANGLE, step_control = NO_DISTANCE_LIMIT),
    reach = (; corrector = CorrectorSettings{Float32}(), step_control = StepControl{Float32}()),
)
const ARMS = (
    summed_kl_angle = (; distance = SummedDistance, ACCEPTANCES.kl_angle...),
    fisher_kl_angle = (; distance = FisherDistance, ACCEPTANCES.kl_angle...),
    summed_reach = (; distance = SummedDistance, ACCEPTANCES.reach...),
    fisher_reach = (; distance = FisherDistance, ACCEPTANCES.reach...),
)
const ARM_WEIGHTS = (0.1f0, 0.5f0, 0.9f0)

"""The `TraceSettings` of `arm` (an `ARMS` entry) at state weight `state_weight`, with `keywords` passed on."""
function arm_settings(arm::NamedTuple, state_weight::E; keywords...)::TraceSettings{E} where {E <: Real}
    geometry = TraceGeometry{E}(; arm.distance, state_weight)
    return TraceSettings{E}(; geometry, arm.corrector, arm.step_control, keywords...)
end

"""Every arm at every weight, as `label => settings` pairs."""
function arm_variants()::Vector{Pair{String, TraceSettings{Float32}}}
    return [string(name, " θ ", weight) => arm_settings(arm, weight) for (name, arm) in pairs(ARMS) for weight in ARM_WEIGHTS]
end
