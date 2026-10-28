using GameTracing
using Test
using BifurcationKit: BifurcationKit
using ErrorTypes: is_error, unwrap, unwrap_error

include("games.jl")
include("fixtures.jl")

# SCRATCH: which fold arm turns back under the residual_scale floor, and where; not for landing.
function main()::Nothing
    arms = (
        default = TraceSettings{Float32}(),
        secant = TraceSettings{Float32}(tangent = BifurcationKit.Secant()),
        bordering = TraceSettings{Float32}(bordered_solver = BorderingSolve),
        uncontrolled = TraceSettings{Float32}(step_control = nothing),
    )
    for (name, settings) in pairs(arms)
        result = trace_endpoint(LogitGame(FOLD_GAME, 2.0f0), 1, settings)
        @info "fold arm" name failed = is_error(result) outcome = is_error(result) ? unwrap_error(result) : unwrap(result)
    end
    for (name, settings) in pairs(arms)
        result = trace_endpoint(LogitGame(FOLD_GAME, 2.0f0), 1, settings)
        if is_error(result)
            @info "debugging arm" name
            ENV["JULIA_DEBUG"] = "GameTracing"
            trace_endpoint(LogitGame(FOLD_GAME, 2.0f0), 1, settings)
            ENV["JULIA_DEBUG"] = ""
        end
    end
    return nothing
end
main()
