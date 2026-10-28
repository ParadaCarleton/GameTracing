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
                # At a floor of 32 ulps of 12 (the fold game's P max|G|), this arm's predictor once jumped to the lower branch.
                TraceSettings{Float32}(
                    bordered_solver = BorderingSolve,
                    corrector = CorrectorSettings{Float32}(; residual_tolerance = 32 * eps(Float32) * 12),
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

@testset "the corrector refuses a predictor that moves a logit past max_step_kl from the step's start" begin
    check = GameTracing.CorrectorReach(
        GameTracing.AcceptAll(),
        CorrectorSettings{Float32}(),
        TraceGeometry{Float32}(),
        zeros(Float32, 2),
        zeros(Float32, 2),
        zero(Float32),
    )
    start = Float32[9.7, 0.46]
    first_iterate(predicted) = (; x = predicted, z0 = (; u = start), step = 0, residual = 1.0f0, res_f = ones(Float32, 2))
    # Logit 4.4 to 4.3 is a small move; logit 4.4 to 0.3 moves 1.2 nats.
    @test check(first_iterate(Float32[9.4, 0.46]); fromNewton = false) === true
    @test check(first_iterate(Float32[0.7, 0.41]); fromNewton = false) === false
end
