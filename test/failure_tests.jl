using ErrorTypes: is_error, unwrap_error
using Logging: Logging
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

"""An `accept_corrected` that refuses every corrector past relative precision `level`."""
struct RefusePast{E <: Real}
    level::E
end

(refuse::RefusePast)(::AbstractVector, relative_precision::Real, ::Bool)::Bool = relative_precision <= refuse.level

# Refused from the flat start, where ρ = 0, and mid-branch.
@testset "halving ends once the predictor cannot move the point ($(typeof(refusal)))" for refusal in (RefuseConverged(), RefusePast(0.25f0))
    logs, result = Test.collect_test_logs(min_level = Logging.Debug) do
        trace_endpoint(LogitGame(GAME, TARGET_PRECISION), WIDTH, FAILURE_SETTINGS; accept_corrected = refusal)
    end
    messages = [(log.message, get(log.kwargs, :reason, nothing)) for log in logs]
    @test unwrap_error(result).cause == GameTracing.TraceStuck
    # At ρ = 0 any move of ρ resolves: halving ends at BifurcationKit's zero step, never at an unresolved predictor.
    @test count(==(("corrector stopped", :unresolved)), messages) == ifelse(refusal isa RefuseConverged, 0, 1)
    # Halving 0.05 to Float32's smallest subnormal 2^-149 takes log2(0.05) + 149 ≈ 145 steps (measured: 143 from
    # the flat start, 41 mid-branch); 15 more allow accepted steps before the refusals.
    @test count(message -> first(message) == "corrector closed", messages) < 160
end

@testset "a predictor that leaves the point unchanged is refused, even after a settled solve" begin
    point = Float32[1, -2, 0.5f0]
    check = GameTracing.CorrectorReach(
        GameTracing.AcceptAll(),
        CorrectorSettings{Float32}(),
        TraceGeometry{Float32}(),
        copy(point),
        copy(point),
        0.0f0,
    )
    GameTracing.restate!(check; settled_here = true)
    @test !check((; x = point .+ eps.(point), z0 = (; u = point), step = 0, residual = 0.0f0); fromNewton = false)
    @test check.state[].unresolved
    @test !check((; x = point, residual = 0.0f0, step = 1, p = 1.0f0, closing = true); fromNewton = false)
    # Each coordinate's move is measured against its own size: any move of a zero coordinate resolves.
    @test !GameTracing.step_unresolved(Float32[0, 1], Float32[1.0f-30, 1])
    @test GameTracing.step_unresolved(Float32[0, 1], Float32[0, 1 + eps(1.0f0)])
    # A large coordinate does not hide a small one's move.
    @test !GameTracing.step_unresolved(Float32[-5.0f6, 1.3f-5], Float32[-5.0f6, 1.4f-5])
    @test !GameTracing.step_unresolved(point, point .+ Float32[0, 0, 1.0f-3])
end

@testset "a corrector reach too short to hold any step ends the trace" begin
    result = trace_endpoint(
        LogitGame(FOLD_GAME, 2.0f0), 1, TraceSettings{Float32}(
            corrector = CorrectorSettings{Float32}(reach = 1.0f-6),
            first_step = 0.05f0,
            max_step = 0.1f0,
        )
    )
    @test is_error(result)
    @test unwrap_error(result).cause in (GameTracing.TraceStuck, GameTracing.NoConvergence)
end

@testset "exceptions BifurcationKit and LAPACK throw are told apart from a caller's" begin
    @test GameTracing.uncorrected_start("Stopping continuation.")
    @test GameTracing.uncorrected_start(
        ErrorException(
            "Newton failed to converge at the start",
        )
    )
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
    solver = GameTracing.bordered_linear_solver(GameTracing.MatrixSolve, 2, Float32)
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
        GameTracing.AcceptAll(),
        CorrectorSettings{Float32}(),
        TraceGeometry{Float32}(),
        previous,
        predicted,
        1.0f0,
    )
    GameTracing.restate!(check; predictor_move = GameTracing.logit_change(previous, predicted))
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
        false,
    )
    result = GameTracing.reached_state(
        progress,
        (; step_length = 0.0f0, accepted_step = 0.0f0, exhausted = nothing, logit_accuracy = 0.0f0, unresolved = false, exception = "Stopping continuation."),
        LogitGame(GAME, TARGET_PRECISION),
        TraceSettings{Float32}(; max_steps = 10),
        (;),
    )
    @test is_error(result)
    @test unwrap_error(result).cause == GameTracing.TraceStuck
end

"""A `trace_record_point` record of BifurcationKit's branch at `step`: the point `(gaps, param)` and a unit tangent."""
function branch_record(
        gaps::AbstractVector{E},
        param::E,
        step::Integer,
    )::NamedTuple where {E <: Real}
    return (; gaps, gap_tangent = zero(gaps), tangent_precision = one(E), param, step)
end

"""`finalise`'s verdict on the accepted `point` (a `branch_record`) of a branch that so far holds `branch`."""
function finalised(
        finalise::GameTracing.TraceFinalise,
        point::NamedTuple,
        branch::AbstractVector,
    )::Bool
    # BifurcationKit calls `finalise` before it saves the point to the branch.
    tangent = (; u = zero([point.gaps; point.param]), p = one(point.param))
    return finalise(
        (; u = [point.gaps; point.param], p = point.param),
        tangent,
        point.step,
        (; branch);
        state = (; ds = one(point.param) / 10),
        iter = (; alg = GameTracing.palc_algorithm(TraceSettings{typeof(point.param)}(), length(point.gaps), TARGET_PRECISION)),
    )
end

@testset "an accepted point equal to the last one stops the trace as a stall" begin
    game = LogitGame(GAME, TARGET_PRECISION)
    finalise = GameTracing.TraceFinalise(
        GameTracing.AcceptAll(),
        GameTracing.NoRedraw(),
        GameTracing.RedrawSlot(game, false, 0.1f0),
        GameTracing.CorrectorSettings{Float32}(),
    )
    start, moved = branch_record(Float32[0, 0], 0.0f0, 0),
        branch_record(Float32[0.1f0, 0.2f0], 0.5f0, 1)
    @test finalised(finalise, moved, [start])
    @test !finalised(finalise, branch_record(moved.gaps, moved.param, 2), [start, moved])
    # A change in the precision alone, or in the logits alone, is progress.
    @test finalised(finalise, branch_record(moved.gaps, 0.6f0, 2), [start, moved])
    @test finalised(
        finalise,
        branch_record(Float32[0.3f0, 0.2f0], moved.param, 2),
        [start, moved],
    )

    opening = GameTracing.TraceProgress(
        zeros(Float32, 2),
        0.0f0,
        0,
        0.0f0,
        0.0f0,
        zeros(Float32, 2),
        0.0f0,
        false,
        false,
    )
    @test GameTracing.branch_progress([start], opening) === opening
    progress = GameTracing.branch_progress([start, moved], opening)
    @test !progress.unmoved
    @test (progress.steps, progress.precision_max) == (1, moved.param)
    @test progress.gaps == moved.gaps && progress.relative_precision == moved.param
    outcome = (; step_length = 0.0f0, accepted_step = 0.0f0, exhausted = nothing, logit_accuracy = 0.0f0, unresolved = false, exception = nothing)
    settings = TraceSettings{Float32}(; max_steps = 2)
    @test unwrap_error(
        GameTracing.reached_state(
            progress,
            outcome,
            game,
            settings,
            (;),
        )
    ).cause ==
        GameTracing.TraceStuck
    stuck = GameTracing.branch_progress(
        [start, moved, branch_record(moved.gaps, moved.param, 2)],
        opening,
    )
    @test stuck.unmoved
    @test stuck.steps == 2
    # A stall that spends the step budget is still a stall; moving points that spend it are not.
    stalled = unwrap_error(GameTracing.reached_state(stuck, outcome, game, settings, (;)))
    @test stalled.cause == GameTracing.TraceStuck
    @test stalled.context.unmoved
    spent = GameTracing.branch_progress(
        [start, moved, branch_record(moved.gaps, 0.6f0, 2)],
        opening,
    )
    @test !spent.unmoved
    @test unwrap_error(
        GameTracing.reached_state(
            spent,
            outcome,
            game,
            settings,
            (;),
        )
    ).cause ==
        GameTracing.NoConvergence
end

"""A branch record at `angle` on the unit circle in the (gap, param) plane, tangent along the circle if `turning`."""
function circle_record(angle::E, step::Integer, turning::Bool)::NamedTuple where {E <: Real}
    tangent_scale = turning ? one(E) : zero(E)
    return (;
        gaps = [cos(angle)], gap_tangent = [-sin(angle) * tangent_scale], tangent_precision = cos(angle) * tangent_scale,
        param = sin(angle), step,
    )
end

"""The records of `laps` laps of `per_lap` steps round the unit circle, from its first point."""
function circle_branch(
        ::Type{E},
        laps::Integer,
        per_lap::Integer,
        turning::Bool,
    )::AbstractVector where {E <: Real}
    return [circle_record(E(2 * pi * (index - 1) / per_lap), index, turning) for index in 1:(laps * per_lap)]
end

"""The 1-based index of the first record of `branch` that `looped` flags against the records before it, or `nothing`."""
function first_loop(branch::AbstractVector)::Union{Nothing, Int}
    return findfirst(
        [
            GameTracing.looped(view(branch, firstindex(branch):(index - 1)), branch[index]) for index in eachindex(branch)
        ]
    )
end

@testset "a branch that goes round a closed loop is flagged $element" for element in (Float32, widen(Float32))
    circle = circle_branch(element, 3, 20, true)
    flagged = first_loop(circle)
    # Fenchel's gate: no flag before the tangents have turned through 2π, which takes 21 records.
    @test !isnothing(flagged) && 21 <= flagged <= 52
    # The gate alone holds back a loop whose tangents never turn; open paths never loop.
    @test isnothing(first_loop(circle_branch(element, 3, 20, false)))
    line = [branch_record([element(index) / 10], element(index) / 20, index) for index in 1:60]
    @test isnothing(first_loop(line))
    # A path that comes within a step of its anchor without its precision ever turning back is no loop.
    rising = [(; record..., tangent_precision = abs(record.tangent_precision) + one(element)) for record in circle]
    @test isnothing(first_loop(rising))
    s_curve = [
        (; branch_record([element(index) / 10], sin(element(index) / 5), index)..., gap_tangent = [one(element)], tangent_precision = cos(element(index) / 5))
            for index in 1:60
    ]
    @test isnothing(first_loop(s_curve))

    game = LogitGame(GAME, TARGET_PRECISION)
    finalise = GameTracing.TraceFinalise(
        GameTracing.AcceptAll(),
        GameTracing.NoRedraw(),
        GameTracing.RedrawSlot(game, false, element(1) / 10),
        GameTracing.CorrectorSettings{typeof(element(1))}(),
    )
    @test finalised(finalise, circle[10], circle[1:9])
    @test !finalised(finalise, circle[flagged], circle[1:(flagged - 1)])

    opening = GameTracing.TraceProgress(zeros(element, 1), zero(element), 0, zero(element), zero(element), zeros(element, 1), zero(element), false, false)
    @test !GameTracing.branch_progress(circle[1:10], opening).looped
    progress = GameTracing.branch_progress(circle[1:flagged], opening)
    @test progress.looped && !progress.unmoved
    failure = unwrap_error(
        GameTracing.reached_state(
            progress,
            (; step_length = zero(element), accepted_step = zero(element), exhausted = nothing, logit_accuracy = zero(element), unresolved = false, exception = nothing),
            game,
            TraceSettings{element}(; max_steps = 1000),
            (;),
        )
    )
    @test failure.cause == GameTracing.TraceStuck
    @test failure.context.looped
end
