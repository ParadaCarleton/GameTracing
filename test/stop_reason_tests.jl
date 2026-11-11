using ErrorTypes: is_error, unwrap_error
using BifurcationKit: BifurcationKit

const STOP_SETTINGS = TraceSettings{Float32}(first_step = 0.01f0, max_step = 0.05f0)

@testset "a non-finite tangent is already a loud Singular failure" begin
    result = trace_endpoint(LogitGame(NaNPrecisionSlopePass(GAME, 0.5f0), TARGET_PRECISION), WIDTH, STOP_SETTINGS)
    @test is_error(result)
    failure = unwrap_error(result)
    @test failure.cause == GameTracing.Singular
    @test failure.context.reason == :solver_blowup
end

"""A trace outcome as `guarded_trace` returns it, stopped with `retcode` after `closings` closed correctors."""
function stopped_outcome(; retcode, closings, predictor_move)
    return (;
        step_length = 0.0f0, accepted_step = 0.0f0, exhausted = nothing, logit_accuracy = 0.0f0, unresolved = false,
        predictor_move, closings, retcode, exception = nothing,
    )
end

@testset "a zero-step stop names its reason ($reason)" for (reason, retcode, closings, predictor_move) in (
        (:nonfinite_predictor, BifurcationKit.ReturnCode.DsLessThanMin, 0, NaN32),
        (:rejected_after_convergence, BifurcationKit.ReturnCode.DsLessThanMin, 3, 0.1f0),
        (:step_floor, BifurcationKit.ReturnCode.DsLessThanMin, 0, 0.1f0),
        (:finalized, BifurcationKit.ReturnCode.Finalized, 1, 0.1f0),
        (:stopped_short, BifurcationKit.ReturnCode.EmptyReturnCode, 0, 0.1f0),
    )
    progress = GameTracing.TraceProgress{Float32}(zeros(Float32, WIDTH), 0.0f0, 0, 0.0f0, 0.0f0, zeros(Float32, WIDTH), 0.0f0, false, false)
    result = GameTracing.reached_state(
        progress,
        stopped_outcome(; retcode, closings, predictor_move),
        LogitGame(GAME, TARGET_PRECISION),
        STOP_SETTINGS,
        (;),
    )
    failure = unwrap_error(result)
    @test failure.context.reason == reason
    @test failure.context.retcode == retcode
    @test failure.cause == (reason == :nonfinite_predictor ? GameTracing.NotFinite : GameTracing.TraceStuck)
end
