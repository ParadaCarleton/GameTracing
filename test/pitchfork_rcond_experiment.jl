# EXPERIMENT: rcond of ∂F/∂w at the exact and unfolded pitchforks' corrector iterates, against eps(Float32).
using BifurcationKit: BifurcationKit
using ErrorTypes: is_error, unwrap_error
using LinearAlgebra: cond
using Logging: Debug
using Printf: @sprintf
using Test: collect_test_logs

"""`game` traced over `width` logits under `settings`, a `Function` as `collect_test_logs` takes."""
struct RcondTrace{G, S} <: Function
    game::G
    width::Int
    settings::S
end

(trace::RcondTrace)()::TraceResult{Float32} = trace_endpoint(trace.game, trace.width, trace.settings)

"""The reciprocal 1-norm condition number of `∂F/∂w` at `point = [w; ρ]` on `game`, in the point's type."""
function gap_rcond(game::LogitGame, point::AbstractVector{E})::E where {E <: Real}
    jacobian = GameTracing.branch_system!(GameTracing.BranchWorkspace(game, point), game, point, last(point)).jacobian
    width = length(point) - 1
    return inv(cond(jacobian[1:width, 1:width], 1))
end

"""The cause a trace's `result` ended with, or `reached`."""
function rcond_ending(result::TraceResult)::String
    if is_error(result)
        return string(unwrap_error(result).cause)
    end
    return "reached"
end

@testset "EXPERIMENT: rcond at the pitchforks" begin
    cases = [
        [(string(name), LOGITNASH_GAMES[name]) for name in PITCHFORK_GAMES];
        [("unfolded 1e-4", UNFOLDED_PITCHFORK_GAME)]
    ]
    for (label, payoffs) in cases, (tangent_label, tangent) in (("bordered", BifurcationKit.Bordered()), ("secant", BifurcationKit.Secant()))
        game = LogitGame(NormalFormPass(map(table -> Float32.(table), payoffs)), 4.0f0)
        logs, result = collect_test_logs(RcondTrace(game, length(payoffs), TraceSettings{Float32}(; tangent)); min_level = Debug)
        iterates = [record.kwargs[:state] for record in logs if record.message == "corrector iterate"]
        rconds = [gap_rcond(game, iterate) for iterate in iterates]
        context = is_error(result) ? unwrap_error(result).context : (; precision = 4.0f0)
        println(@sprintf(
            "RCOND %s %s: %s at lambda %.5g; last iterate rcond %.3g, smallest %.3g over %d iterates; eps %.3g",
            label, tangent_label, rcond_ending(result), context.precision, last(rconds), minimum(rconds), length(rconds), eps(Float32),
        ))
        @test !isempty(rconds)
    end
end
