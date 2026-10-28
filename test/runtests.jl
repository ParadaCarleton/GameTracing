using GameTracing
using Test
using ErrorTypes: is_error, unwrap, unwrap_error
using Random: Xoshiro

include("games.jl")
include("gambit.jl")

# SCRATCH: log probability_bound at the last compared point of each far-target game; not for landing.
const FAR_OPENING_SEED = 0xb0bf341028e24ba6
const MIXED_STALL_SEED = 0x233ce38d667063a1
const FAR_TARGET = 1.0f6
const LOG_PRECISION_SETTINGS = TraceSettings{Float32}(; geometry = TraceGeometry{Float32}(; distance = FisherLogPrecisionDistance))

far_games(rng) = [ntuple(_ -> 3 .* randn(rng, Float32, counts), length(counts)) for counts in ((3, 3), (2, 2, 2))]

struct StopAt{E <: Real}
    precision::E
end
(stop::StopAt)(::AbstractVector, relative_precision::Real)::Bool = relative_precision * FAR_TARGET < stop.precision

function report(label, payoffs, stop_precision)::Nothing
    gains = NormalFormGains(payoffs)
    failure = unwrap_error(trace_endpoint(LogitGame(gains, FAR_TARGET), GameTracing.logit_count(gains), LOG_PRECISION_SETTINGS; keep_going = StopAt(Float32(stop_precision))))
    (; precision, logits) = failure.context
    corrector = LOG_PRECISION_SETTINGS.corrector
    with_scale = GameTracing.corrector_tolerance(corrector.residual_tolerance, corrector.tolerance_floor_ulps, [logits; 1.0f0], GameTracing.residual_scale(LogitGame(gains, precision)))
    gaps_only = GameTracing.corrector_tolerance(corrector.residual_tolerance, corrector.tolerance_floor_ulps, [logits; 1.0f0], 0.0f0)
    oracle = gambit_strategies(payoffs, precision)
    gap = maximum(maximum.(abs, GameTracing.mixed_strategies(gains, logits) .- oracle.strategies))
    @info "far-target bound" label stop_precision precision gap residual_with_scale = with_scale bound_with_scale = probability_bound(gains, logits, precision, with_scale, 0.5f0) residual_gaps_only = gaps_only bound_gaps_only = probability_bound(gains, logits, precision, gaps_only, 0.5f0)
    return nothing
end

function main()::Nothing
    seeded = (far_games(Xoshiro(FAR_OPENING_SEED))..., last(far_games(Xoshiro(MIXED_STALL_SEED))))
    for (index, payoffs) in pairs(seeded), stop in (50, 100)
        report("seeded $index", payoffs, stop)
    end
    rng = Xoshiro(0x5eed)
    for index in 1:4, (shape, payoffs) in pairs(far_games(rng)), stop in (50, 100)
        report("drawn $index.$shape", payoffs, stop)
    end
    return nothing
end
main()
