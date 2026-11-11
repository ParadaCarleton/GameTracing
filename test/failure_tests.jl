using ErrorTypes: is_error, unwrap_error
using Logging: Logging
using LinearAlgebra: LAPACKException, SingularException, norm

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
        LogitGame(ThrowingPass(GAME, ArgumentError("bad game"), 0.0f0), TARGET_PRECISION),
        WIDTH,
        FAILURE_SETTINGS,
    )
end

@testset "a LAPACK failure mid-trace is a failure value at the precision it reached" begin
    result = trace_endpoint(
        LogitGame(ThrowingPass(GAME, SingularException(1), 1.0f0), TARGET_PRECISION),
        WIDTH,
        FAILURE_SETTINGS,
    )
    @test is_error(result)
    @test unwrap_error(result).cause == GameTracing.Singular
    @test 0 < TARGET_PRECISION * unwrap_error(result).context.relative_precision <= 1.0f0
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

"""The testset "a predictor that leaves the point unchanged is refused, even after a settled solve", in the element type `E`."""
function check_unresolved_predictor(::Type{E})::Nothing where {E <: Real}
    point = E[1, -2, inv(E(2))]
    check = GameTracing.CorrectorReach(
        GameTracing.AcceptAll(), nothing,
        CorrectorSettings{E}(),
        TraceGeometry{E}(),
        GameTracing.PalcBorder(TraceGeometry{E}(), 2, one(E)),
        copy(point),
        copy(point),
        zero(E),
    )
    GameTracing.restate!(check; settled_here = true)
    @test !check((; x = point .+ eps.(point), z0 = (; u = point), step = 0, residual = zero(E)); fromNewton = false)
    @test check.state[].unresolved
    @test !check((; x = point, residual = zero(E), step = 1, p = one(E), closing = true); fromNewton = false)
    ulps = CorrectorSettings{E}().resolution_ulps
    # Each coordinate's move is measured against its own size: any move of a zero coordinate resolves.
    @test !GameTracing.step_unresolved(E[0, 1], E[floatmin(E), 1], ulps)
    @test GameTracing.step_unresolved(E[0, 1], E[0, 1 + eps(one(E))], ulps)
    # A large coordinate does not hide a small one's move.
    @test !GameTracing.step_unresolved(E[-5_000_000, E(13) / 1_000_000], E[-5_000_000, E(14) / 1_000_000], ulps)
    @test !GameTracing.step_unresolved(point, point .+ E[0, 0, inv(E(1000))], ulps)
    # The resolution is a setting: a move of a few ulps resolves under a finer one.
    @test GameTracing.step_unresolved(E[0, 1], E[0, 1 + 2 * eps(one(E))], ulps)
    @test !GameTracing.step_unresolved(E[0, 1], E[0, 1 + 2 * eps(one(E))], one(E))
    return nothing
end

@testset "a predictor that leaves the point unchanged is refused, even after a settled solve ($element)" for element in (Float32, widen(Float32))
    check_unresolved_predictor(element)
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
    # A step that halves to the float type's resolution after reach refusals is labeled PrecisionExhausted
    # (`unresolved_noise`): a refused correction under `logit_tolerance` is rounding noise too. The trace still fails.
    @test unwrap_error(result).cause in (GameTracing.TraceStuck, GameTracing.NoConvergence, GameTracing.PrecisionExhausted)
    @test isfinite(unwrap_error(result).context.relative_precision)
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
        GameTracing.AcceptAll(), nothing,
        CorrectorSettings{Float32}(),
        TraceGeometry{Float32}(),
        GameTracing.PalcBorder(TraceGeometry{Float32}(), 2, 1.0f0),
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
        GameTracing.AcceptAll(), nothing,
        CorrectorSettings{Float32}(),
        TraceGeometry{Float32}(),
        GameTracing.PalcBorder(TraceGeometry{Float32}(), 1, 1.0f0),
        previous,
        predicted,
        1.0f0,
    )
    GameTracing.restate!(check; predictor_move = GameTracing.logit_change(previous, predicted))
    @test GameTracing.within_reach(check, predicted .+ Float32[1.0f-2, 0])
    @test !GameTracing.within_reach(check, predicted .+ Float32[3, 0])
end

# The Newton system's last row at fixed ρ, in a two-action game's `[w; ρ]`.
pin_row(::Type{E}) where {E <: Real} = E[0, 0, 1]

"""`GAME` in the element type `E`."""
game_in(::Type{E}) where {E <: Real} = LogitGame(LinearGainPass(E.(GAME.interaction), E.(GAME.stand_alone)), E(TARGET_PRECISION))

"""A `CorrectorReach` on `game` in PALC's default geometry, its step from `previous` to `predicted`."""
function rounding_check(game, settings::CorrectorSettings{E}, previous, predicted)::GameTracing.CorrectorReach where {E <: Real}
    return GameTracing.CorrectorReach(
        GameTracing.AcceptAll(), game, settings, TraceGeometry{E}(),
        GameTracing.PalcBorder(TraceGeometry{E}(), length(previous) - 1, E(TARGET_PRECISION)), previous, predicted, zero(E),
    )
end

"""The testset "a stalled correction within the rounding bound settles, and one past it does not", in the element type `E`."""
function check_rounding_settle(::Type{E})::Nothing where {E <: Real}
    game = game_in(E)
    point = E[0, 0, E(3) / 2]
    floor = GameTracing.rounding_correction(game, point, pin_row(E))
    # The bound is rounding of an O(1) residual through an O(1) inverse: a few ulps.
    @test 0 < floor <= 1000 * eps(one(E))
    @test GameTracing.rounding_correction(nothing, point, pin_row(E)) == 0
    # A row that makes J singular has no bound: the floor is then `logit_tolerance` alone.
    @test GameTracing.rounding_correction(game, point, zeros(E, 3)) == 0
    settings = CorrectorSettings{E}(logit_tolerance = floor / 1000)
    # Two corrections of one floor each in logits: the second fails to contract.
    iterates = [point .+ E[step * floor / last(point), 0, 0] for step in 0:2]
    # The margin is a setting: under a quarter of the bound the same stall does not settle.
    for (noise_game, margin, settles) in ((game, 4, true), (nothing, 4, false), (game, 1 // 4, false))
        check = rounding_check(noise_game, CorrectorSettings{E}(; logit_tolerance = settings.logit_tolerance, noise_margin = E(margin)), point, point)
        @test GameTracing.corrected!(check, (; x = iterates[1], fx = zeros(E, 3), step = 0), true)
        @test GameTracing.corrected!(check, (; x = iterates[2], fx = zeros(E, 3), step = 1), true) === true
        stalled = GameTracing.corrected!(check, (; x = iterates[3], fx = zeros(E, 3), step = 2), true)
        @test (stalled isa GameTracing.BifurcationKit.NewtonConverged) == settles
        @test check.state[].settled_here == settles
    end
    return nothing
end

@testset "a stalled correction within the rounding bound settles, and one past it does not ($element)" for element in (Float32, widen(Float32))
    check_rounding_settle(element)
end

"""The testset "a PALC corrector settles at rounding only once its step has stalled at noise", in the element type `E`."""
function check_palc_rounding_settle(::Type{E})::Nothing where {E <: Real}
    game = game_in(E)
    point = E[0, 0, E(3) / 2]
    floor = GameTracing.rounding_correction(game, point, pin_row(E))
    # A predictor along ρ alone: its arclength row is the pin row, so the bound is the fixed-ρ one.
    predicted = point .+ E[0, 0, inv(E(1000))]
    settings = CorrectorSettings{E}(logit_tolerance = floor / 1000)
    @test GameTracing.rounding_correction(game, point, GameTracing.bordering_row(rounding_check(game, settings, point, predicted))) ≈ floor
    iterates = [point .+ E[step * floor / last(point), 0, 0] for step in 0:2]
    # A first stall halves the step; the stall at its halved step settles.
    earlier_stall = (; relative_precision = last(point), correction = floor, confirmed = false)
    for (suspect, settles) in ((nothing, false), (earlier_stall, true))
        check = rounding_check(game, settings, point, predicted)
        GameTracing.restate!(check; predictor_move = floor, noise = GameTracing.NoiseRecord(zero(E), suspect))
        @test GameTracing.corrected!(check, (; x = iterates[1], fx = zeros(E, 3), step = 0), true)
        @test GameTracing.corrected!(check, (; x = iterates[2], fx = zeros(E, 3), step = 1), true) === true
        stalled = GameTracing.corrected!(check, (; x = iterates[3], fx = zeros(E, 3), step = 2), true)
        @test (stalled isa GameTracing.BifurcationKit.NewtonConverged) == settles
        @test check.state[].settled_here == settles
    end
    return nothing
end

@testset "a PALC corrector settles at rounding only once its step has stalled at noise ($element)" for element in (Float32, widen(Float32))
    check_palc_rounding_settle(element)
end

"""The testset "the rounding bound's arclength row is PALC's, in PALC's weights", in the element type `E`."""
function check_weighted_row(::Type{E})::Nothing where {E <: Real}
    geometry = TraceGeometry{E}(; distance = FisherLogPrecisionDistance)
    target = E(1_000_000)
    previous, predicted = E[2, -3, inv(E(100_000))], E[E(5) / 2, -E(7) / 2, E(12) / 1_000_000]
    palc = GameTracing.palc_geometry(geometry, 2, target)
    check = GameTracing.CorrectorReach(
        GameTracing.AcceptAll(), nothing, CorrectorSettings{E}(), geometry, GameTracing.PalcBorder(palc), previous, predicted, zero(E),
    )
    GameTracing.restate!(check; predictor_move = GameTracing.logit_change(previous, predicted))
    row = GameTracing.bordering_row(check)
    direction = predicted .- previous
    @test row ≈ GameTracing.palc_border(palc.dot, palc.palc_state_weight, direction)
    # PALC's product weighs `ρ` far above the gaps at a far target: the raw step is another row.
    @test !(row / norm(row) ≈ direction / norm(direction))
    # PALC resets its Fisher weights in place at each accepted point, and the row reads them as they stand.
    GameTracing.BifurcationKit.reweight!(palc.dot, GameTracing.BifurcationKit.BorderedArray(E[40, -60, inv(E(100_000))], inv(E(100_000))))
    @test GameTracing.bordering_row(check) ≈ GameTracing.palc_border(palc.dot, palc.palc_state_weight, direction)
    @test !(GameTracing.bordering_row(check) ≈ row)
    return nothing
end

@testset "the rounding bound's arclength row is PALC's, in PALC's weights ($element)" for element in (Float32, widen(Float32))
    check_weighted_row(element)
end

@testset "PALC and the corrector's rounding bound share one dotθ" begin
    game = LogitGame(GAME, TARGET_PRECISION)
    settings = TraceSettings{Float32}(; geometry = TraceGeometry{Float32}(; distance = FisherLogPrecisionDistance))
    start = Float32[0, 0, 0]
    palc = GameTracing.palc_geometry(settings.geometry, WIDTH, TARGET_PRECISION)
    corrector = GameTracing.CorrectorReach(
        GameTracing.AcceptAll(), game, settings.corrector, settings.geometry, GameTracing.PalcBorder(palc), start, start, zero(Float32),
    )
    finalise = GameTracing.TraceFinalise(
        GameTracing.AcceptAll(), GameTracing.NoRedraw(), GameTracing.RedrawSlot(game, false, 0.1f0), settings.corrector,
    )
    iterator = GameTracing.trace_iterator(game, start, settings, palc, finalise, corrector, 0.1f0, 10)
    @test iterator.alg.dotθ === corrector.border.dot
end

"""The testset "near a fold the arclength row bounds the rounding, where the fixed-ρ bound grows without limit", in the element type `E`."""
function check_fold_bound(::Type{E})::Nothing where {E <: Real}
    # FOLD_GAME's logit solves z = λ 6 σ(z - 3), so λ(z) = z (1 + exp(3 - z)) / 6 folds where λ'(z) = 0.
    target = E(2)
    game = LogitGame(ThresholdPass(E(FOLD_GAME.amplitude), E(FOLD_GAME.threshold)), target)
    precision_at(logit) = logit * (1 + exp(3 - logit)) / 6
    slope(logit) = (1 + exp(3 - logit) - logit * exp(3 - logit)) / 6
    low, high = one(E), E(3)
    for _ in 1:100
        middle = (low + high) / 2
        (low, high) = slope(low) * slope(middle) <= 0 ? (low, middle) : (middle, high)
    end
    fold = (low + high) / 2
    branch_point(logit) = [logit / (precision_at(logit) / target), precision_at(logit) / target]
    for geometry in (TraceGeometry{E}(), TraceGeometry{E}(; distance = FisherLogPrecisionDistance))
        border = GameTracing.PalcBorder(geometry, 1, target)
        for distance in inv.(E.((1_000, 10_000, 100_000)))
            logit = fold + distance
            point = branch_point(logit)
            tangent = branch_point(logit + inv(E(1000))) .- branch_point(logit - inv(E(1000)))
            fixed = GameTracing.rounding_correction(game, point, E[0, 1])
            bordered = GameTracing.rounding_correction(game, point, border(tangent))
            # Measured in Float32: the fixed-ρ bound is about 0.3 ulp / distance (0.04 at 1e-5); the bordered one stays
            # near 3 ulps.
            @test fixed >= eps(one(E)) / 10 / distance
            @test 0 < bordered <= 10 * eps(one(E))
        end
    end
    return nothing
end

@testset "near a fold the arclength row bounds the rounding, where the fixed-ρ bound grows without limit ($element)" for element in (Float32, widen(Float32))
    check_fold_bound(element)
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
    # A stalled record has no direction, so it turns nothing.
    stalled = (; line[1]..., gap_tangent = [zero(element)], tangent_precision = zero(element))
    @test GameTracing.tangent_turn(stalled, line[2]) == 0
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
