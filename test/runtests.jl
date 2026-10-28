using GameTracing
using Test
using BifurcationKit: BifurcationKit
using ErrorTypes: is_error, unwrap, unwrap_error

include("games.jl")
include("fixtures.jl")

# SCRATCH: accepted steps with max_angle off and at the π/2 default, per fixed game; not for landing.
function steps(result)::String
    return is_error(result) ? "FAILED $(unwrap_error(result).cause)" : string(unwrap(result).steps)
end

function main()::Nothing
    angles = (off = nothing, right = Float32(π) / 2)
    fold_arms = (
        default = TraceSettings{Float32},
        secant = (; corrector) -> TraceSettings{Float32}(; corrector, tangent = BifurcationKit.Secant()),
        bordering = (; corrector) -> TraceSettings{Float32}(; corrector, bordered_solver = BorderingSolve),
        uncontrolled = (; corrector) -> TraceSettings{Float32}(; corrector, step_control = nothing),
    )
    for (arm, make) in pairs(fold_arms), (label, max_angle) in pairs(angles)
        settings = make(; corrector = CorrectorSettings{Float32}(; max_angle))
        println("fold $arm angle=$label steps=", steps(trace_endpoint(LogitGame(FOLD_GAME, 2.0f0), 1, settings)))
    end
    for (name, payoffs) in pairs(LOGITNASH_GAMES), precision in (4.0f0, 20.0f0), (label, max_angle) in pairs(angles)
        settings = TraceSettings{Float32}(; corrector = CorrectorSettings{Float32}(; max_angle))
        pass = NormalFormPass(map(table -> Float32.(table), payoffs))
        println("game $name λ=$precision angle=$label pitchfork=$(name in PITCHFORK_GAMES) steps=", steps(trace_endpoint(LogitGame(pass, precision), 2, settings)))
    end
    return nothing
end
main()
