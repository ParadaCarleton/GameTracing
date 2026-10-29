using ErrorTypes: is_error, unwrap_error
using LinearAlgebra: LAPACKException, SingularException

const FAILURE_SETTINGS = TraceSettings{Float32}(first_step = 0.01f0, max_step = 0.05f0)

@testset "a non-finite response past a precision stops the trace short of it" begin
    result = trace_endpoint(
        LogitGame(PoisonedPass(GAME, 1.0f0), TARGET_PRECISION),
        WIDTH,
        FAILURE_SETTINGS,
    )
    @test is_error(result)
    @test unwrap_error(result).cause == GameTracing.TraceStuck
    @test TARGET_PRECISION * unwrap_error(result).context.relative_precision <= 1.0f0
end

@testset "a non-finite Jacobian is a failure value" begin
    result = trace_endpoint(
        LogitGame(NaNSlopePass(GAME), TARGET_PRECISION),
        WIDTH,
        FAILURE_SETTINGS,
    )
    @test is_error(result)
    @test unwrap_error(result).cause in (GameTracing.Singular, GameTracing.TraceStuck)
end

@testset "a caller's exception propagates" begin
    @test_throws ArgumentError trace_endpoint(
        LogitGame(ThrowingPass(GAME), TARGET_PRECISION),
        WIDTH,
        FAILURE_SETTINGS,
    )
end

struct RefuseConverged end

(::RefuseConverged)(::AbstractVector, ::Real, ::Bool)::Bool = false

@testset "refusing every corrector ends the trace" begin
    result = trace_endpoint(
        LogitGame(GAME, TARGET_PRECISION),
        WIDTH,
        FAILURE_SETTINGS;
        accept_corrected = RefuseConverged(),
    )
    @test is_error(result)
    @test unwrap_error(result).cause == GameTracing.TraceStuck
end

@testset "a corrector reach too short to hold any step ends the trace" begin
    result = trace_endpoint(LogitGame(FOLD_GAME, 2.0f0), 1, TraceSettings{Float32}(
        corrector = CorrectorSettings{Float32}(reach = 1.0f-6),
        first_step = 0.05f0,
        max_step = 0.1f0,
    );
    )
    @test is_error(result)
    @test unwrap_error(result).cause in (GameTracing.TraceStuck, GameTracing.NoConvergence)
end

@testset "exceptions BifurcationKit and LAPACK throw are told apart from a caller's" begin
    @test GameTracing.uncorrected_start("Stopping continuation.")
    @test GameTracing.uncorrected_start(ErrorException(
        "Newton failed to converge at the start",
    ))
    @test !GameTracing.uncorrected_start(ErrorException("another"))
    @test !GameTracing.uncorrected_start("another")
    @test !GameTracing.uncorrected_start(ArgumentError("bad game"))
    @test GameTracing.solver_blowup(SingularException(2))
    @test GameTracing.solver_blowup(LAPACKException(2))
    @test GameTracing.solver_blowup(ArgumentError("matrix contains Infs or NaNs"))
    @test GameTracing.solver_blowup(ArgumentError("LAPACK failed"))
    @test !GameTracing.solver_blowup(ArgumentError("bad game"))
    @test !GameTracing.solver_blowup(DimensionMismatch("bad game"))
end

@testset "a bordered matrix of the wrong size is rejected" begin
    @test_throws DimensionMismatch GameTracing.bordered_matrix!(
        zeros(Float32, 2, 2),
        ones(Float32, 2, 2),
        ones(Float32, 2),
        ones(Float32, 2),
        1.0f0,
        1.0f0,
        1.0f0,
        nothing,
    )
end

@testset "a triangular bordered system solves like the dense one" begin
    solver = GameTracing.matrix_bls(2, Float32)
    jacobian = Float32[2 0; 0 3]
    border = Float32[0, 0]
    solution, parameter, _, _ = solver(
        jacobian,
        border,
        Float32[0, 0],
        1.0f0,
        Float32[2, 3],
        1.0f0,
    )
    @test collect(solution) ≈ Float32[1, 1]
    @test parameter ≈ 1.0f0
end

@testset "a predictor that did not move has no reach to exceed" begin
    check = GameTracing.CorrectorReach(
        GameTracing.AcceptAll(),
        CorrectorSettings{Float32}(),
        TraceGeometry{Float32}(),
        zeros(Float32, 3),
        zeros(Float32, 3),
        0.0f0,
    )
    @test GameTracing.within_reach(check, ones(Float32, 3))
end

@testset "reach measures logits: a far target's gap rounding is within it, a logit jump is not" begin
    # At ρ = 1e-5 a gap move of 1e-2, a far target's rounding of w = P G, moves the logit 1e-7.
    previous, predicted = Float32[0, 1.0f-5], Float32[1, 1.0f-5]
    check = GameTracing.CorrectorReach(
        GameTracing.AcceptAll(), CorrectorSettings{Float32}(), TraceGeometry{Float32}(), previous, predicted, 1.0f0,
    )
    check.predictor_move = GameTracing.logit_change(previous, predicted)
    @test GameTracing.within_reach(check, predicted .+ Float32[1.0f-2, 0])
    @test !GameTracing.within_reach(check, predicted .+ Float32[3, 0])
end

@testset "a game that is non-finite at every positive precision fails at the first step" begin
    result = trace_endpoint(
        LogitGame(PoisonedPass(GAME, 0.0f0), TARGET_PRECISION),
        WIDTH,
        FAILURE_SETTINGS,
    )
    @test is_error(result)
    @test unwrap_error(result).cause == GameTracing.TraceStuck
    @test unwrap_error(result).context.steps == 0
end

@testset "BifurcationKit's report of an uncorrectable start is a stuck trace" begin
    progress = GameTracing.TraceProgress(
        zeros(Float32, 2),
        0.0f0,
        0,
        0.0f0,
        0.0f0,
        zeros(Float32, 2),
        0.0f0,
        false,
    )
    result = GameTracing.reached_state(
        progress,
        (; step_length = 0.0f0, exhausted = nothing, logit_accuracy = 0.0f0, exception = "Stopping continuation."),
        LogitGame(GAME, TARGET_PRECISION),
        TraceSettings{Float32}(; max_steps = 10),
        (;),
    )
    @test is_error(result)
    @test unwrap_error(result).cause == GameTracing.TraceStuck
end

"""A `trace_record_point` record of BifurcationKit's branch at `step`: the point `(gaps, param)` and a unit tangent."""
branch_record(gaps::AbstractVector{E}, param::E, step::Integer)::NamedTuple where {E <: Real} = (;
    gaps, gap_tangent = zero(gaps), tangent_precision = one(E), param, step,
)

"""`finalise`'s verdict on the accepted `point` (a `branch_record`) of a branch that so far holds `branch`."""
function finalised(finalise::GameTracing.TraceFinalise, point::NamedTuple, branch::AbstractVector)::Bool
    # BifurcationKit calls `finalise` before it saves the point to the branch.
    tangent = (; u = zero([point.gaps; point.param]), p = one(point.param))
    return finalise((; u = [point.gaps; point.param], p = point.param), tangent, point.step, (; branch); state = (; ds = 0.1f0))
end

@testset "an accepted point equal to the last one stops the trace as a stall" begin
    game = LogitGame(GAME, TARGET_PRECISION)
    finalise = GameTracing.TraceFinalise(GameTracing.AcceptAll(), GameTracing.NoRedraw(), GameTracing.RedrawSlot(game, game, false, 0.1f0))
    start, moved = branch_record(Float32[0, 0], 0.0f0, 0), branch_record(Float32[0.1f0, 0.2f0], 0.5f0, 1)
    @test finalised(finalise, moved, [start])
    @test !finalised(finalise, branch_record(moved.gaps, moved.param, 2), [start, moved])
    # A change in the precision alone, or in the logits alone, is progress.
    @test finalised(finalise, branch_record(moved.gaps, 0.6f0, 2), [start, moved])
    @test finalised(finalise, branch_record(Float32[0.3f0, 0.2f0], moved.param, 2), [start, moved])

    opening = GameTracing.TraceProgress(zeros(Float32, 2), 0.0f0, 0, 0.0f0, 0.0f0, zeros(Float32, 2), 0.0f0, false)
    @test GameTracing.branch_progress([start], opening) === opening
    progress = GameTracing.branch_progress([start, moved], opening)
    @test !progress.unmoved
    @test (progress.steps, progress.precision_max) == (1, moved.param)
    outcome = (; step_length = 0.0f0, exhausted = nothing, logit_accuracy = 0.0f0, exception = nothing)
    settings = TraceSettings{Float32}(; max_steps = 2)
    @test unwrap_error(GameTracing.reached_state(progress, outcome, game, settings, (;))).cause == GameTracing.TraceStuck
    stuck = GameTracing.branch_progress([start, moved, branch_record(moved.gaps, moved.param, 2)], opening)
    @test stuck.unmoved
    @test stuck.steps == 2
    # A stall that spends the step budget is still a stall; moving points that spend it are not.
    stalled = unwrap_error(GameTracing.reached_state(stuck, outcome, game, settings, (;)))
    @test stalled.cause == GameTracing.TraceStuck
    @test stalled.context.unmoved
    spent = GameTracing.branch_progress([start, moved, branch_record(moved.gaps, 0.6f0, 2)], opening)
    @test !spent.unmoved
    @test unwrap_error(GameTracing.reached_state(spent, outcome, game, settings, (;))).cause == GameTracing.NoConvergence
end
