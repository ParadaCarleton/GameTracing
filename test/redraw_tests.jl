using ErrorTypes: is_error, unwrap, unwrap_error
using LinearAlgebra: norm, opnorm
using Logging: Debug
using Test: collect_test_logs

const REDRAW_SETTINGS = TraceSettings{Float32}()

"""The trace of `swap` that redraws its game from `before` to `after` at `swap.swap_level`."""
function swapped_trace(swap::MapSwap)::TraceResult{Float32}
    return trace_endpoint(
        LogitGame(swap.before, swap.precision),
        swap.width,
        REDRAW_SETTINGS;
        redraw = SwapAt(LogitGame(swap.after, swap.precision), swap.swap_level),
    )
end

"""`(; logits, relative_precision)` of the accepted point at which `swapped_trace(swap)` redraws: the same trace, stopped there."""
function swap_point(swap::MapSwap)::NamedTuple
    stopped = trace_endpoint(
        LogitGame(swap.before, swap.precision),
        swap.width,
        REDRAW_SETTINGS;
        keep_going = StopAt(swap.swap_level),
    )
    (; logits, relative_precision) = unwrap_error(stopped).context
    return (; logits, relative_precision)
end

@testset "a redraw mid-trace: the endpoint solves the new map, on the branch a double-precision reference follows from the swap" begin
    result = swapped_trace(NEAR_SWAP)
    @test !is_error(result)
    logits = unwrap(result).logits
    # The corrector's sup-norm residual in gaps equals the logit residual at ρ = 1; 2x covers recomputing it.
    tolerance = GameTracing.corrector_tolerance(
        REDRAW_SETTINGS.corrector.residual_tolerance,
        REDRAW_SETTINGS.corrector.tolerance_floor_ulps,
        logits,
    )
    @test residual_norm(NEAR_SWAP.after, logits, NEAR_SWAP.precision) < 2 * tolerance
    swap = swap_point(NEAR_SWAP)
    @test swap.relative_precision >= NEAR_SWAP.swap_level
    reference = branch_reference(
        LinearGainPass(NEAR_SWAP.after.interaction .* float(1), NEAR_SWAP.after.stand_alone .* float(1)),
        NEAR_SWAP.width,
        NEAR_SWAP.precision;
        start = swap.logits .* float(1),
        start_precision = swap.relative_precision * NEAR_SWAP.precision,
    )
    @test reference.reached
    @test reference.crossings == 0
    # Probabilities to 1e-3, as the LogitNash cross-check compares a Float32 trace with its reference.
    @test maximum(abs, inv.(1 .+ exp.(-logits)) .- reference.first_choices) < 1.0f-3
    # The swap moved the branch: the endpoint is not the old map's.
    @test residual_norm(NEAR_SWAP.before, logits, NEAR_SWAP.precision) > 100 * tolerance
    @info "bytes per redraw" swapped = @allocated(swapped_trace(NEAR_SWAP)) unswapped = @allocated(
        trace_endpoint(LogitGame(NEAR_SWAP.before, NEAR_SWAP.precision), NEAR_SWAP.width, REDRAW_SETTINGS)
    )
end

"""Whether the logits of `result` solve `pass` at `precision` to twice the corrector tolerance at `ρ = 1`."""
function on_map(result::TraceResult{Float32}, pass::Any, precision::Real)::Bool
    logits = unwrap(result).logits
    tolerance = GameTracing.corrector_tolerance(
        REDRAW_SETTINGS.corrector.residual_tolerance,
        REDRAW_SETTINGS.corrector.tolerance_floor_ulps,
        [logits; 1.0f0],
    )
    return residual_norm(pass, logits, precision) < 2 * tolerance
end

@testset "a redraw whose homotopy rises to the new map's only root crosses there, 3.5 logits on" begin
    swap = swap_point(FAR_SWAP)
    @test rising_homotopy(FAR_SWAP.before, FAR_SWAP.after, swap.relative_precision * FAR_SWAP.precision, only(swap.logits))
    result = swapped_trace(FAR_SWAP)
    @test !is_error(result)
    @test on_map(result, FAR_SWAP.after, FAR_SWAP.precision)
end

@testset "a redraw whose homotopy folds before the new map fails stuck and projected, at the fold" begin
    swap = swap_point(FOLD_SWAP)
    @test !rising_homotopy(FOLD_SWAP.before, FOLD_SWAP.after, swap.relative_precision * FOLD_SWAP.precision, only(swap.logits))
    result = swapped_trace(FOLD_SWAP)
    @test is_error(result)
    failure = unwrap_error(result)
    @test failure.cause == GameTracing.TraceStuck
    @test failure.context.projected
    @test failure.context.reason == :fold
    @test failure.context.relative_precision == swap.relative_precision
    @test failure.context.step_length > 0
    # The failure keeps the accepted point on the old map, not a root of the new one.
    @test residual_norm(FOLD_SWAP.before, failure.context.logits, failure.context.precision) < 1.0f-3
end

"""`game` traced over `width` logits with `redraw`, a `Function` as `collect_test_logs` takes."""
struct LoggedTrace{G, R} <: Function
    game::G
    width::Int
    redraw::R
end

(trace::LoggedTrace)()::TraceResult{Float32} = trace_endpoint(trace.game, trace.width, REDRAW_SETTINGS; redraw = trace.redraw)

"""`swap`'s redrawn trace as a `LoggedTrace`."""
function logged_swap(swap::MapSwap)::LoggedTrace
    redraw = SwapAt(LogitGame(swap.after, swap.precision), swap.swap_level)
    return LoggedTrace(LogitGame(swap.before, swap.precision), swap.width, redraw)
end

"""The `reason` of each "trace stopped" record that `trace` logs, and its result."""
function stop_reasons(trace::LoggedTrace)::NamedTuple
    logs, result = collect_test_logs(trace; min_level = Debug)
    return (; reasons = [record.kwargs[:reason] for record in logs if record.message == "trace stopped"], result)
end

@testset "a trace that stops short logs why, once: a redraw that folds, a non-finite game; a reached trace logs no stop" begin
    folded = stop_reasons(logged_swap(FOLD_SWAP))
    @test unwrap_error(folded.result).context.projected
    @test folded.reasons == [:fold]
    poisoned = stop_reasons(LoggedTrace(LogitGame(PoisonedPass(GAME, 1.0f0), TARGET_PRECISION), WIDTH, GameTracing.NoRedraw()))
    @test unwrap_error(poisoned.result).cause == GameTracing.TraceStuck
    @test length(poisoned.reasons) == 1
    @test GameTracing.STOP_CAUSES[only(poisoned.reasons)] == GameTracing.TraceStuck
    near = stop_reasons(logged_swap(NEAR_SWAP))
    @test !is_error(near.result)
    @test isempty(near.reasons)
end

@testset "a redraw at the target precision ends on a root of the new map" begin
    swap = MapSwap(NEAR_SWAP.before, NEAR_SWAP.after, NEAR_SWAP.precision, 1.0f0, NEAR_SWAP.width)
    result = swapped_trace(swap)
    @test !is_error(result)
    logits = unwrap(result).logits
    tolerance = GameTracing.corrector_tolerance(
        REDRAW_SETTINGS.corrector.residual_tolerance,
        REDRAW_SETTINGS.corrector.tolerance_floor_ulps,
        [logits; 1.0f0],
    )
    @test residual_norm(swap.after, logits, swap.precision) < 2 * tolerance
    @test residual_norm(swap.before, logits, swap.precision) > 100 * tolerance
end

"""
    probability_bound(pass, logits, precision, residual)

The largest first-action probability error of a root of `pass` at `precision` whose logit residual is `residual`, to
first order: the logit error `‖J⁻¹‖∞ residual`, `J = I - ∂(λG)/∂z` at the Float64 root `logits`, plus Float32's 32-ulp
rounding of the logits, times the logistic's largest slope 1/4.
"""
function probability_bound(pass::Any, logits::AbstractVector, precision::Real, residual::Real)::typeof(float(residual))
    jacobian = reference_system(pass, [logits; precision * float(1)]).jacobian[:, 1:(end - 1)]
    logit_error = opnorm(inv(jacobian), Inf32) * residual + 32 * eps(Float32) * maximum(abs, logits)
    return logit_error / 4
end

@testset "every endpoint solves the map in effect at the target, whatever the swap level" begin
    rng = fresh_rng()
    levels = [rand(rng, Float32, 8); 1.0f0]
    for level in levels
        swap = MapSwap(NEAR_SWAP.before, NEAR_SWAP.after, NEAR_SWAP.precision, level, NEAR_SWAP.width)
        result = swapped_trace(swap)
        @test !is_error(result)
        @test on_map(result, swap.after, swap.precision)
    end
    # A swap on the target point when that point spends the step budget: the budget the unswapped trace took.
    budget = unwrap(trace_endpoint(LogitGame(NEAR_SWAP.before, NEAR_SWAP.precision), NEAR_SWAP.width, REDRAW_SETTINGS)).steps
    @test budget > 0
    last_step = trace_endpoint(
        LogitGame(NEAR_SWAP.before, NEAR_SWAP.precision),
        NEAR_SWAP.width,
        TraceSettings{Float32}(; max_steps = budget);
        redraw = SwapAt(LogitGame(NEAR_SWAP.after, NEAR_SWAP.precision), 1.0f0),
    )
    @test !is_error(last_step)
    @test on_map(last_step, NEAR_SWAP.after, NEAR_SWAP.precision)
end

@testset "an endpoint at the target that misses its game's tolerance fails OffMap" begin
    game = LogitGame(GAME, TARGET_PRECISION)
    traced = unwrap(trace_endpoint(game, WIDTH, REDRAW_SETTINGS))
    progress = GameTracing.TraceProgress(copy(traced.logits), 1.0f0, 3, 0.0f0, 1.0f0, zeros(Float32, WIDTH), 0.0f0, false)
    outcome = (; step_length = 0.1f0, exception = nothing)
    @test unwrap(GameTracing.reached_state(progress, outcome, game, REDRAW_SETTINGS, (;))).steps == 3
    moved = LogitGame(LinearGainPass(GAME.interaction, -GAME.stand_alone), TARGET_PRECISION)
    failure = unwrap_error(GameTracing.reached_state(progress, outcome, moved, REDRAW_SETTINGS, (;)))
    @test failure.cause == GameTracing.OffMap
    @test failure.context.residual >= failure.context.tolerance
end

@testset "a redrawn game must keep the trace's target precision" begin
    game = LogitGame(GAME, TARGET_PRECISION)
    slot = GameTracing.RedrawSlot(game, game, false, 0.1f0)
    @test !GameTracing.swapped!(slot, nothing)
    @test !GameTracing.swapped!(slot, slot.game)
    @test !slot.pending
    @test_throws ArgumentError GameTracing.swapped!(slot, LogitGame(GAME, 2 * TARGET_PRECISION))
    @test GameTracing.swapped!(slot, LogitGame(LinearGainPass(GAME.interaction, -GAME.stand_alone), TARGET_PRECISION))
    @test slot.pending
    @test slot.before === game
end

@testset "a redraw of random swap 6 at relative precision 0.734 crosses to the branch a double-precision reference follows" begin
    (; swap, reference_after) = random_swap(6)
    # A Newton correction moved this swap's logits 0.439 from the old root, past a fixed bound of 0.4.
    level = 0.734f0
    stopped = trace_endpoint(LogitGame(swap.before, swap.precision), swap.width, REDRAW_SETTINGS; keep_going = StopAt(level))
    (; logits, relative_precision) = unwrap_error(stopped).context
    @test relative_precision >= level
    point = [logits ./ relative_precision; relative_precision]
    crossing = GameTracing.redraw_crossing(LogitGame(swap.before, swap.precision), LogitGame(swap.after, swap.precision), point, REDRAW_SETTINGS)
    @test crossing.reason == :crossed
    @test crossing.steps > 0
    crossed = GameTracing.logits_at(crossing.point)
    @test residual_norm(swap.after, crossed, relative_precision * swap.precision) < 1.0f-3
    @test norm(crossed .- logits) > 0.4f0
    @info "crossing steps" swap = "swap 6" crossing.steps
    result = trace_endpoint(LogitGame(swap.before, swap.precision), swap.width, REDRAW_SETTINGS; redraw = SwapAt(LogitGame(swap.after, swap.precision), level))
    @test !is_error(result)
    endpoint = unwrap(result).logits
    reference = branch_reference(reference_after, swap.width, swap.precision; start = logits .* float(1), start_precision = relative_precision * swap.precision)
    @test reference.reached
    @test reference.crossings == 0
    # The endpoint's residual is under the corrector tolerance at ρ = 1, where gap and logit residuals agree; 2x covers recomputing it.
    residual = 2 * GameTracing.corrector_tolerance(REDRAW_SETTINGS.corrector.residual_tolerance, REDRAW_SETTINGS.corrector.tolerance_floor_ulps, [endpoint; 1.0f0])
    @test maximum(abs, inv.(1 .+ exp.(-endpoint)) .- reference.first_choices) < probability_bound(reference_after, reference.logits, swap.precision, residual)
end

@testset "a redraw onto a game that is non-finite at the redraw fails stuck and projected, as a solver blowup" begin
    # The new game is non-finite past precision 1, so its branch has no points at the redraw's precision 2.
    before = PoisonedPass(NEAR_SWAP.before, Inf32)
    after = PoisonedPass(NEAR_SWAP.after, 1.0f0)
    result = trace_endpoint(
        LogitGame(before, NEAR_SWAP.precision),
        NEAR_SWAP.width,
        REDRAW_SETTINGS;
        redraw = SwapAt(LogitGame(after, NEAR_SWAP.precision), NEAR_SWAP.swap_level),
    )
    @test is_error(result)
    failure = unwrap_error(result)
    @test failure.cause == GameTracing.TraceStuck
    @test failure.context.projected
    @test failure.context.reason == :solver_blowup
end

@testset "a redraw whose homotopy passes a pitchfork with no fold fails stuck and projected, at the crossing" begin
    swap = swap_point(PITCHFORK_SWAP)
    level = swap.relative_precision * PITCHFORK_SWAP.precision
    (; before, after) = PITCHFORK_SWAP
    # The blend's interaction runs from the old map's to the new one's; sampled in `s`, only the antisymmetric mode changes sign.
    modes = [
        symmetric_modes((1 - blend) * before.interaction[1, 2] + blend * after.interaction[1, 2], level)
        for blend in range(0, 1; length = 101)
    ]
    @test all(mode.symmetric > 0 for mode in modes)
    @test first(modes).antisymmetric > 0 > last(modes).antisymmetric
    result = swapped_trace(PITCHFORK_SWAP)
    @test is_error(result)
    failure = unwrap_error(result)
    @test failure.cause == GameTracing.TraceStuck
    @test failure.context.projected
    @test failure.context.reason == :crossing
end

@testset "a redraw crossing rethrows an exception it does not convert" begin
    swap = swap_point(NEAR_SWAP)
    point = [swap.logits ./ swap.relative_precision; swap.relative_precision]
    before = LogitGame(NEAR_SWAP.before, NEAR_SWAP.precision)
    after = LogitGame(ThrowingPass(NEAR_SWAP.after, 0.0f0), NEAR_SWAP.precision)
    @test_throws DomainError GameTracing.redraw_crossing(before, after, point, REDRAW_SETTINGS)
end
