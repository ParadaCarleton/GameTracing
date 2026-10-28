using GameTracing
using ErrorTypes: is_error, unwrap, unwrap_error
using Random: Xoshiro

# SCRATCH: print the 3x3 draws and their traced endpoints at 1e3 for a local pygambit comparison; not for landing.
function main()::Nothing
    rng = Xoshiro(0x5eed)
    for _ in 1:6
        ntuple(_ -> 3 .* randn(rng, Float32, 2, 2, 2), 3)
    end
    for index in 1:4
        payoffs = ntuple(_ -> 3 .* randn(rng, Float32, 3, 3), 2)
        gains = NormalFormGains(payoffs)
        for (label, max_step_kl) in ((:off, nothing), (:half, 0.5f0))
            settings = TraceSettings{Float32}(; corrector = CorrectorSettings{Float32}(; max_step_kl))
            result = trace_endpoint(LogitGame(gains, 1.0f3), 4, settings)
            profile = is_error(result) ? "FAILED $(unwrap_error(result).cause)" : repr(mixed_strategies(gains, unwrap(result).logits))
            println("DRAW33 $index $label steps=", is_error(result) ? -1 : unwrap(result).steps, " profile=", profile)
        end
        println("PAYOFFS33 $index ", repr(map(table -> vec(table), payoffs)))
    end
    return nothing
end
main()
