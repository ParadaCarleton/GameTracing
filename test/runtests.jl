using GameTracing
using Test
using ErrorTypes: is_error, unwrap, unwrap_error
using Random: Xoshiro

include("games.jl")
include("fixtures.jl")

# SCRATCH: the KL and angle of every closing point, per trace, under default settings with the angle check off; not for landing.
function main()::Nothing
    plain = TraceSettings{Float32}(; corrector = CorrectorSettings{Float32}(; max_angle = nothing))
    println("TRACE fold-jump")
    loose = TraceSettings{Float32}(; bordered_solver = BorderingSolve, corrector = CorrectorSettings{Float32}(; max_angle = nothing, residual_tolerance = 32 * eps(Float32) * 12))
    trace_endpoint(LogitGame(FOLD_GAME, 2.0f0), 1, loose)
    println("TRACE fold")
    trace_endpoint(LogitGame(FOLD_GAME, 2.0f0), 1, plain)
    for (name, payoffs) in pairs(LOGITNASH_GAMES)
        if !(name in PITCHFORK_GAMES) && size(first(payoffs)) == (2, 2)
            println("TRACE $name")
            trace_endpoint(LogitGame(NormalFormPass(map(table -> Float32.(table), payoffs)), 20.0f0), 2, plain)
        end
    end
    rng = Xoshiro(0x5eed)
    for index in 1:6
        payoffs = ntuple(_ -> 3 .* randn(rng, Float32, 2, 2, 2), 3)
        println("TRACE drawn $index")
        trace_endpoint(LogitGame(NormalFormGains(payoffs), 1.0f3), 3, plain)
    end
    return nothing
end
main()
