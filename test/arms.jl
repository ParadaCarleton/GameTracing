# The trace-geometry arms in payoff gaps: step distance (Fisher or log precision) with the shipped reach acceptance,
# each at three state weights.
const ARMS = (
    fisher_reach = (; distance = FisherDistance),
    log_precision_reach = (; distance = LogPrecisionDistance),
)
const ARM_WEIGHTS = (0.01f0, 0.03f0, 0.1f0)

"""The `TraceSettings` of `arm` (an `ARMS` entry) at state weight `state_weight`, with `keywords` passed on."""
function arm_settings(arm::NamedTuple, state_weight::E; keywords...)::TraceSettings{E} where {E <: Real}
    geometry = TraceGeometry{E}(; arm.distance, state_weight)
    return TraceSettings{E}(; geometry, keywords...)
end

"""Every arm at every weight, as `label => settings` pairs."""
function arm_variants()::Vector{Pair{String, TraceSettings{Float32}}}
    return [string(name, " θ ", weight) => arm_settings(arm, weight) for (name, arm) in pairs(ARMS) for weight in ARM_WEIGHTS]
end
