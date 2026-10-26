# The trace-geometry arms: coordinates, step distance and acceptance, each at three state weights.
# Arms 1-3 accept by KL and chord angle with the reach and first-step distance tests off; arm 4 keeps them.
const KL_ANGLE = CorrectorSettings{Float32}(reach = nothing, max_kl = 0.01f0, max_angle = 0.5f0)
const NO_DISTANCE_LIMIT = StepControl{Float32}(distance_limit = nothing)
const ARMS = (
    arm1 = (; coordinates = PayoffGaps, distance = SummedDistance, corrector = KL_ANGLE, step_control = NO_DISTANCE_LIMIT),
    arm2 = (; coordinates = Logits, distance = SummedDistance, corrector = KL_ANGLE, step_control = NO_DISTANCE_LIMIT),
    arm3 = (; coordinates = PayoffGaps, distance = FisherDistance, corrector = KL_ANGLE, step_control = NO_DISTANCE_LIMIT),
    arm4 = (; coordinates = PayoffGaps, distance = SummedDistance, corrector = CorrectorSettings{Float32}(), step_control = StepControl{Float32}()),
)
const ARM_WEIGHTS = (0.1f0, 0.5f0, 0.9f0)

"""The `TraceSettings` of `arm` (an `ARMS` entry) at state weight `state_weight`, with `keywords` passed on."""
function arm_settings(arm::NamedTuple, state_weight::E; keywords...)::TraceSettings{E} where {E <: Real}
    geometry = TraceGeometry{E}(; arm.coordinates, arm.distance, state_weight)
    return TraceSettings{E}(; geometry, arm.corrector, arm.step_control, keywords...)
end

"""Every arm at every weight, as `label => settings` pairs."""
function arm_variants()::Vector{Pair{String, TraceSettings{Float32}}}
    return [string(name, " θ ", weight) => arm_settings(arm, weight) for (name, arm) in pairs(ARMS) for weight in ARM_WEIGHTS]
end
