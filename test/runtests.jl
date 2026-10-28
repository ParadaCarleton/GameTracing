using GameTracing
using ErrorTypes: is_error, unwrap, unwrap_error

include("games.jl")
include("fixtures.jl")

# SCRATCH: both KL directions at every predictor and closing point of the fold arms; not for landing.
function main()::Nothing
    loose = 32 * eps(Float32) * 12
    for (label, max_step_kl) in ((:off, nothing), (:half, 0.5f0))
        println("TRACE fold-jump bound=$label")
        settings = TraceSettings{Float32}(; bordered_solver = BorderingSolve, corrector = CorrectorSettings{Float32}(; max_step_kl, residual_tolerance = loose))
        result = trace_endpoint(LogitGame(FOLD_GAME, 2.0f0), 1, settings)
        println("RESULT ", is_error(result) ? unwrap_error(result) : unwrap(result).logits)
    end
    return nothing
end
main()
