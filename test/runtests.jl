using GameTracing
using Random: Xoshiro

# SCRATCH: λ and profile at both ends of each 3x3 step with chord KL above 0.1, for a pygambit check; not for landing.
struct OutlierLog{G}
    gains::G
    index::Int
    previous::Vector{Float32}
    previous_precision::Vector{Float32}
end
function (logbook::OutlierLog)(logits::AbstractVector, relative_precision::Real)::Bool
    chord = maximum(GameTracing.bernoulli_kl.(logits, logbook.previous))
    if chord > 0.1f0
        println("OUTLIER $(logbook.index) kl=$chord lam_before=", 1.0f3 * only(logbook.previous_precision), " before=", repr(mixed_strategies(logbook.gains, logbook.previous)),
            " lam_after=", 1.0f3 * relative_precision, " after=", repr(mixed_strategies(logbook.gains, collect(logits))))
    end
    logbook.previous .= logits
    logbook.previous_precision .= relative_precision
    return true
end

function main()::Nothing
    rng = Xoshiro(0x5eed)
    for _ in 1:6
        ntuple(_ -> 3 .* randn(rng, Float32, 2, 2, 2), 3)
    end
    for index in 1:4
        gains = NormalFormGains(ntuple(_ -> 3 .* randn(rng, Float32, 3, 3), 2))
        trace_endpoint(LogitGame(gains, 1.0f3), 4, TraceSettings{Float32}(); keep_going = OutlierLog(gains, index, zeros(Float32, 4), zeros(Float32, 1)))
    end
    return nothing
end
main()
