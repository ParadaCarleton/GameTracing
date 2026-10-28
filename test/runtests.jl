using GameTracing
using Test
using ErrorTypes: is_error, unwrap, unwrap_error
using Random: Xoshiro

include("games.jl")
include("fixtures.jl")

# SCRATCH: the largest per-logit KL between consecutive accepted points (the chord), per trace; not for landing.
struct ChordLog
    previous::Vector{Float32}
end
function (logbook::ChordLog)(logits::AbstractVector, ::Real)::Bool
    println("chord kl=", maximum(GameTracing.bernoulli_kl.(logits, logbook.previous)))
    logbook.previous .= logits
    return true
end

function main()::Nothing
    plain = TraceSettings{Float32}(; corrector = CorrectorSettings{Float32}(; max_angle = nothing))
    println("TRACE fold-jump")
    loose = TraceSettings{Float32}(; bordered_solver = BorderingSolve, corrector = CorrectorSettings{Float32}(; max_angle = nothing, residual_tolerance = 32 * eps(Float32) * 12))
    trace_endpoint(LogitGame(FOLD_GAME, 2.0f0), 1, loose; keep_going = ChordLog(zeros(Float32, 1)))
    println("TRACE fold")
    trace_endpoint(LogitGame(FOLD_GAME, 2.0f0), 1, plain; keep_going = ChordLog(zeros(Float32, 1)))
    for (name, payoffs) in pairs(LOGITNASH_GAMES)
        if !(name in PITCHFORK_GAMES) && size(first(payoffs)) == (2, 2)
            println("TRACE $name")
            trace_endpoint(LogitGame(NormalFormPass(map(table -> Float32.(table), payoffs)), 20.0f0), 2, plain; keep_going = ChordLog(zeros(Float32, 2)))
        end
    end
    rng = Xoshiro(0x5eed)
    for index in 1:6
        payoffs = ntuple(_ -> 3 .* randn(rng, Float32, 2, 2, 2), 3)
        println("TRACE drawn $index")
        trace_endpoint(LogitGame(NormalFormGains(payoffs), 1.0f3), 3, plain; keep_going = ChordLog(zeros(Float32, 3)))
    end
    for index in 1:4
        payoffs = ntuple(_ -> 3 .* randn(rng, Float32, 3, 3), 2)
        println("TRACE drawn33 $index")
        trace_endpoint(LogitGame(NormalFormGains(payoffs), 1.0f3), 4, plain; keep_going = ChordLog(zeros(Float32, 4)))
    end
    return nothing
end
main()
