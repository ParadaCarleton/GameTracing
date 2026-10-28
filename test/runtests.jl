using GameTracing
using Test
using ErrorTypes: is_error, unwrap, unwrap_error

include("games.jl")
include("fixtures.jl")

# SCRATCH: which max_angle or max_kl refuses the fold jump, and its step cost on the fixed games; not for landing.
function steps(result)::String
    return is_error(result) ? "FAILED $(unwrap_error(result).cause)" : string(unwrap(result).steps)
end

function main()::Nothing
    limits = (
        right = (; max_angle = Float32(π) / 2),
        quarter = (; max_angle = Float32(π) / 4),
        eighth = (; max_angle = Float32(π) / 8),
        sixteenth = (; max_angle = Float32(π) / 16),
        kl_half = (; max_angle = nothing, max_kl = 0.5f0),
        kl_tenth = (; max_angle = nothing, max_kl = 0.1f0),
        kl_hundredth = (; max_angle = nothing, max_kl = 0.01f0),
    )
    loose = 32 * eps(Float32) * 12
    for (label, limit) in pairs(limits)
        settings = TraceSettings{Float32}(; bordered_solver = BorderingSolve, corrector = CorrectorSettings{Float32}(; residual_tolerance = loose, limit...))
        println("fold-jump $label steps=", steps(trace_endpoint(LogitGame(FOLD_GAME, 2.0f0), 1, settings)))
        plain = TraceSettings{Float32}(; corrector = CorrectorSettings{Float32}(; limit...))
        println("fold $label steps=", steps(trace_endpoint(LogitGame(FOLD_GAME, 2.0f0), 1, plain)))
        for (name, payoffs) in pairs(LOGITNASH_GAMES)
            if !(name in PITCHFORK_GAMES) && size(first(payoffs)) == (2, 2)
                pass = NormalFormPass(map(table -> Float32.(table), payoffs))
                println("game $name $label steps=", steps(trace_endpoint(LogitGame(pass, 20.0f0), 2, plain)))
            end
        end
    end
    return nothing
end
main()
