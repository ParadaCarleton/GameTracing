using GameTracing
using Test
using BifurcationKit: BifurcationKit
using ErrorTypes: is_error, unwrap, unwrap_error
using Random: Xoshiro

include("games.jl")
include("fixtures.jl")
include("gambit.jl")

# SCRATCH: steps with max_step_kl off and at 0.5; Gambit gaps of the draws whose largest step was 0.34 and 0.22; not for landing.
function steps(result)::String
    return is_error(result) ? "FAILED $(unwrap_error(result).cause)" : string(unwrap(result).steps)
end

function main()::Nothing
    bounds = (off = nothing, half = 0.5f0)
    makers = (
        default = (corrector) -> TraceSettings{Float32}(; corrector),
        secant = (corrector) -> TraceSettings{Float32}(; corrector, tangent = BifurcationKit.Secant()),
        bordering = (corrector) -> TraceSettings{Float32}(; corrector, bordered_solver = BorderingSolve),
        uncontrolled = (corrector) -> TraceSettings{Float32}(; corrector, step_control = nothing),
    )
    for (arm, make) in pairs(makers), (label, max_step_kl) in pairs(bounds)
        println("fold $arm bound=$label steps=", steps(trace_endpoint(LogitGame(FOLD_GAME, 2.0f0), 1, make(CorrectorSettings{Float32}(; max_step_kl)))))
    end
    loose = 32 * eps(Float32) * 12
    for (label, max_step_kl) in pairs(bounds)
        settings = TraceSettings{Float32}(; bordered_solver = BorderingSolve, corrector = CorrectorSettings{Float32}(; max_step_kl, residual_tolerance = loose))
        println("fold-jump bound=$label steps=", steps(trace_endpoint(LogitGame(FOLD_GAME, 2.0f0), 1, settings)))
    end
    for (name, payoffs) in pairs(LOGITNASH_GAMES), (label, max_step_kl) in pairs(bounds), precision in (4.0f0, 20.0f0)
        if !(name in PITCHFORK_GAMES) && size(first(payoffs)) == (2, 2)
            settings = TraceSettings{Float32}(; corrector = CorrectorSettings{Float32}(; max_step_kl))
            pass = NormalFormPass(map(table -> Float32.(table), payoffs))
            println("game $name λ=$precision bound=$label steps=", steps(trace_endpoint(LogitGame(pass, precision), 2, settings)))
        end
    end
    rng = Xoshiro(0x5eed)
    drawn = [ntuple(_ -> 3 .* randn(rng, Float32, 2, 2, 2), 3) for _ in 1:6]
    drawn33 = [ntuple(_ -> 3 .* randn(rng, Float32, 3, 3), 2) for _ in 1:4]
    for (label, max_step_kl) in pairs(bounds), (index, payoffs) in pairs([drawn; drawn33])
        settings = TraceSettings{Float32}(; corrector = CorrectorSettings{Float32}(; max_step_kl))
        gains = NormalFormGains(payoffs)
        result = trace_endpoint(LogitGame(gains, 1.0f3), GameTracing.logit_count(gains), settings)
        oracle = gambit_strategies(payoffs, 1.0f3).strategies
        gap = is_error(result) ? NaN32 : maximum(maximum.(abs, GameTracing.mixed_strategies(gains, unwrap(result).logits) .- oracle))
        println("drawn $index bound=$label steps=", steps(result), " gambit_gap=", gap)
    end
    return nothing
end
main()
