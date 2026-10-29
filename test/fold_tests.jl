using BifurcationKit: BifurcationKit
using ErrorTypes: is_error, unwrap
const FOLD_SETTINGS = TraceSettings{Float32}()

@testset "a trace crosses a fold of the branch" begin
    # At target 2 the game has one equilibrium, past a fold at precision about 1.4: only continuation through the fold reaches it.
    target = 2.0f0
    result = trace_endpoint(LogitGame(FOLD_GAME, target), 1, FOLD_SETTINGS)
    @test !is_error(result)
    @test residual_norm(FOLD_GAME, unwrap(result).logits, target) < 1.0f-3
    @test only(unwrap(result).logits) > 3
    @testset "every tangent and solver choice crosses it" begin
        for settings in (
                TraceSettings{Float32}(tangent = BifurcationKit.Secant()),
                TraceSettings{Float32}(
                    bordered_solver = BorderingSolve,
                ),
                TraceSettings{Float32}(step_control = nothing),
                # Past the second fold this arm's Fisher-weighted tangent once turned back (`PlainOrientation`).
                TraceSettings{Float32}(
                    bordered_solver = BorderingSolve,
                    corrector = CorrectorSettings{Float32}(; logit_tolerance = 1.0f-5),
                ),
            )
            traced = trace_endpoint(LogitGame(FOLD_GAME, target), 1, settings)
            @test !is_error(traced)
            @test only(unwrap(traced).logits) ≈ only(unwrap(result).logits) rtol = 1.0f-3
        end
    end
end

@testset "below the fold the trace stays on the lower branch" begin
    # Precision 1 has three equilibria; the branch connected to the flat start is the lowest.
    result = trace_endpoint(LogitGame(FOLD_GAME, 1.0f0), 1, FOLD_SETTINGS)
    @test !is_error(result)
    @test residual_norm(FOLD_GAME, unwrap(result).logits, 1.0f0) < 1.0f-3
    @test only(unwrap(result).logits) < 2
end

@testset "the corrector refuses a closing point that moves a logit past max_step_kl from the step's start" begin
    start = [9.7f0, 0.46f0]
    # The predictor sits at the corrected point, so only the step KL can refuse it.
    accepted(corrected) = GameTracing.closing_accepted(
        GameTracing.CorrectorReach(GameTracing.AcceptAll(), CorrectorSettings{Float32}(), TraceGeometry{Float32}(), start, copy(corrected), 1.0f0),
        corrected,
    )
    # Logit 4.46 to 4.32 is a small move; logit 4.46 to 0.29 moves KL 1.2 nats.
    @test accepted([9.4f0, 0.46f0])
    @test !accepted([0.7f0, 0.41f0])
end
