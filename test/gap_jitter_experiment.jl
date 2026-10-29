# EXPERIMENT: the noise in the gaps w = P G along far-target traces, and the share of each step's arclength in w.
using BifurcationKit: BifurcationKit
using ErrorTypes: is_error, unwrap_error
using Logging: Debug
using Printf: @sprintf
using Random: AbstractRNG, Xoshiro
using Test: collect_test_logs

const JITTER_TARGET = 1.0f6
const JITTER_POINTS = 50
# The default geometry (FisherDistance); the log-precision run is commit 57ebec7.
const JITTER_SETTINGS = TraceSettings{Float32}()

"""Normal-form games with strategy counts `(3, 3)` and `(2, 2, 2)`, payoffs `3 randn`, as the opening tests draw them."""
function jitter_games(rng::AbstractRNG)::AbstractVector
    return [ntuple(_ -> 3 .* randn(rng, Float32, counts), length(counts)) for counts in ((3, 3), (2, 2, 2))]
end

"""`game` traced over `width` logits under `settings`, a `Function` as `collect_test_logs` takes."""
struct JitterTrace{G, S} <: Function
    game::G
    width::Int
    settings::S
end

(trace::JitterTrace)()::TraceResult{Float32} = trace_endpoint(trace.game, trace.width, trace.settings)

"""
The largest change of the gaps between consecutive corrector iterates past a solve's first correction (step 2 on),
in gaps and in logits, from the "corrector iterate" records.
"""
function late_corrections(records::AbstractVector)::NamedTuple
    iterates = [record.kwargs for record in records if record.message == "corrector iterate"]
    consecutive = [
        (earlier[:state], later[:state])
        for (earlier, later) in zip(iterates[1:(end - 1)], iterates[2:end])
        if later[:step] == earlier[:step] + 1 && later[:step] >= 2
    ]
    gaps = [maximum(abs, GameTracing.payoff_gaps(later) .- GameTracing.payoff_gaps(earlier)) for (earlier, later) in consecutive]
    logits = [GameTracing.logit_change(earlier, later) for (earlier, later) in consecutive]
    return (; gaps, logits)
end

"""Shares of PALC's squared step length in the gaps and in ρ, from `from` to `to`, under the trace's dot reweighted at `from`."""
function arclength_shares(from::AbstractVector{E}, to::AbstractVector{E}, geometry::NamedTuple)::NTuple{2, E} where {E <: Real}
    BifurcationKit.reweight!(geometry.dot, BifurcationKit.BorderedArray(from, last(from)))
    in_gaps_step = [GameTracing.payoff_gaps(to .- from); zero(E)]
    in_precision_step = [zero(GameTracing.payoff_gaps(to)); last(to) - last(from)]
    in_gaps = GameTracing.palc_product(geometry.dot, geometry.palc_state_weight, in_gaps_step, in_gaps_step)
    in_precision = GameTracing.palc_product(geometry.dot, geometry.palc_state_weight, in_precision_step, in_precision_step)
    return (in_gaps / (in_gaps + in_precision), in_precision / (in_gaps + in_precision))
end

"""`min / median / max` of `values`, or a dash when there are none."""
function spread(values::AbstractVector)::String
    if isempty(values)
        return "-"
    end
    sorted = sort(values)
    return @sprintf("%.3g / %.3g / %.3g, zero in %d of %d", first(sorted), sorted[cld(length(sorted), 2)], last(sorted), count(iszero, sorted), length(sorted))
end

"""The cause a trace's `result` ended with, or `reached`."""
function ending(result::TraceResult)::String
    if is_error(result)
        return string(unwrap_error(result).cause)
    end
    return "reached"
end

@testset "EXPERIMENT: gap noise and arclength share along far-target traces" begin
    games = [jitter_games(Xoshiro(0xb0bf341028e24ba6))..., last(jitter_games(Xoshiro(0x233ce38d667063a1)))]
    for (label, payoffs) in zip(("far 3x3", "far 2x2x2", "stall 2x2x2"), games)
        gains = NormalFormGains(payoffs)
        width = GameTracing.logit_count(gains)
        logs, result = collect_test_logs(JitterTrace(LogitGame(gains, JITTER_TARGET), width, JITTER_SETTINGS); min_level = Debug)
        points = [record.kwargs[:state] for record in logs if record.message == "trace point"]
        geometry = GameTracing.palc_geometry(JITTER_SETTINGS.geometry, width, JITTER_TARGET)
        scale = JITTER_TARGET * eps(Float32) * maximum(maximum(abs, table) for table in payoffs)
        corrections = late_corrections(logs)
        println("JITTER $label: $(ending(result)), $(length(points)) points, P eps max|payoff| = $(@sprintf("%.3g", scale))")
        println("JITTER $label late corrections in gaps: $(spread(corrections.gaps))")
        println("JITTER $label late corrections in logits: $(spread(corrections.logits))")
        for (index, (from, to)) in enumerate(zip(points[1:(end - 1)], points[2:end]))
            if index > JITTER_POINTS
                break
            end
            gaps, precision = arclength_shares(from, to, geometry)
            println(@sprintf(
                "JITTER %s point %3d lambda %.4g dw %.3g drho %.3g share_w %.3f share_rho %.3f",
                label, index, JITTER_TARGET * last(to), maximum(abs, GameTracing.payoff_gaps(to .- from)), last(to) - last(from), gaps, precision,
            ))
        end
        @test !isempty(points)
    end
end
