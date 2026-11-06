using ErrorTypes: is_error, unwrap, unwrap_error
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
    @test settled_logits(NEAR_SWAP.after, logits, NEAR_SWAP.precision, REDRAW_SETTINGS)
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
    @test newton_correction(NEAR_SWAP.before, logits, NEAR_SWAP.precision) > 10 * REDRAW_SETTINGS.corrector.logit_tolerance
    @info "bytes per redraw" swapped = @allocated(swapped_trace(NEAR_SWAP)) unswapped = @allocated(
        trace_endpoint(LogitGame(NEAR_SWAP.before, NEAR_SWAP.precision), NEAR_SWAP.width, REDRAW_SETTINGS)
    )
end

"""Whether the logits of `result` solve `pass` at `precision` as the corrector settles them (`settled_logits`)."""
function on_map(result::TraceResult{Float32}, pass::Any, precision::Real)::Bool
    return settled_logits(pass, unwrap(result).logits, precision, REDRAW_SETTINGS)
end

"""
Whether `swap`'s redrawn trace failed `TraceStuck` at its swap point, on a root of the old map: the step after a
redraw that moves the root past the corrector's reach halves until no step is left, and no point of the new map is
accepted.
"""
function stuck_at_swap(swap::MapSwap)::Bool
    point = swap_point(swap)
    result = swapped_trace(swap)
    if !is_error(result)
        return false
    end
    failure = unwrap_error(result)
    return failure.cause == GameTracing.TraceStuck && failure.context.relative_precision == point.relative_precision &&
        residual_norm(swap.before, failure.context.logits, failure.context.precision) < 1.0f-3
end

@testset "a redraw that moves the root 3.5 logits fails stuck at the swap, on the old map" begin
    swap = swap_point(FAR_SWAP)
    # The homotopy reaches the new map's only root, but no corrector from the swap point does.
    @test rising_homotopy(FAR_SWAP.before, FAR_SWAP.after, swap.relative_precision * FAR_SWAP.precision, only(swap.logits))
    @test stuck_at_swap(FAR_SWAP)
end

@testset "a redraw past a fold of the homotopy fails stuck at the swap, on the old map" begin
    swap = swap_point(FOLD_SWAP)
    @test !rising_homotopy(FOLD_SWAP.before, FOLD_SWAP.after, swap.relative_precision * FOLD_SWAP.precision, only(swap.logits))
    @test stuck_at_swap(FOLD_SWAP)
end

"""The `RootShift` verdict on a candidate game at the first accepted point a `ShiftProbe` sees from its swap level."""
struct ShiftVerdict <: Exception
    passes::Bool
end

"""A `redraw` hook that throws the `ShiftVerdict` on `game` at the first accepted point from `swap_level` on."""
struct ShiftProbe{G, E <: Real}
    game::G
    swap_level::E
end

function (probe::ShiftProbe)(::AbstractVector, relative_precision::Real, shift::GameTracing.RootShift)::Nothing
    if relative_precision >= probe.swap_level
        throw(ShiftVerdict(shift(probe.game).within))
    end
    return nothing
end

"""Whether `swap`'s redraw passes the `RootShift` check at its swap point."""
function shift_passes(swap::MapSwap)::Bool
    try
        trace_endpoint(
            LogitGame(swap.before, swap.precision), swap.width, REDRAW_SETTINGS;
            redraw = ShiftProbe(LogitGame(swap.after, swap.precision), swap.swap_level)
        )
    catch verdict
        if verdict isa ShiftVerdict
            return verdict.passes
        end
        rethrow()
    end
    error("the trace never reached the swap level")
end

@testset "the root-shift check passes the near swap and refuses the far, fold and pitchfork swaps" begin
    @test shift_passes(NEAR_SWAP)
    @test !shift_passes(FAR_SWAP)
    @test !shift_passes(FOLD_SWAP)
    @test !shift_passes(PITCHFORK_SWAP)
    off = TraceSettings{Float32}(; corrector = GameTracing.CorrectorSettings{Float32}(; reach = nothing))
    @test GameTracing.RootShift(Float32[1, 1], Float32[0.1f0, 0.1f0], Float32[0, 1], off.corrector)(LogitGame(FAR_SWAP.after, 1.0f0)).within
end

@testset "a root shift is a Newton step onto the new map: it cuts the new map's correction by a factor of ten" begin
    swap = swap_point(NEAR_SWAP)
    point = [swap.logits ./ swap.relative_precision; swap.relative_precision]
    after = LogitGame(NEAR_SWAP.after, NEAR_SWAP.precision)
    shift = GameTracing.RootShift(point, Float32[0, 0, 1], nothing)(after).shift
    @test last(shift) == 0
    level = swap.relative_precision * NEAR_SWAP.precision
    before_shift = newton_correction(NEAR_SWAP.after, swap.logits, level)
    @test before_shift > 10 * REDRAW_SETTINGS.corrector.logit_tolerance
    @test newton_correction(NEAR_SWAP.after, GameTracing.logits_at(point .+ shift), level) < before_shift / 10
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

@testset "a trace that stops short logs why, once: a redraw past the reach, a non-finite game; a reached trace logs no stop" begin
    far = stop_reasons(logged_swap(FAR_SWAP))
    @test unwrap_error(far.result).cause == GameTracing.TraceStuck
    @test length(far.reasons) == 1
    poisoned = stop_reasons(LoggedTrace(LogitGame(PoisonedPass(GAME, 1.0f0), TARGET_PRECISION), WIDTH, GameTracing.NoRedraw()))
    @test unwrap_error(poisoned.result).cause == GameTracing.TraceStuck
    @test length(poisoned.reasons) == 1
    @test GameTracing.STOP_CAUSES[only(poisoned.reasons)] == GameTracing.TraceStuck
    near = stop_reasons(logged_swap(NEAR_SWAP))
    @test !is_error(near.result)
    @test isempty(near.reasons)
end

@testset "a redraw requested at the target is not installed: the endpoint solves the map it was traced on" begin
    swap = MapSwap(NEAR_SWAP.before, NEAR_SWAP.after, NEAR_SWAP.precision, 1.0f0, NEAR_SWAP.width)
    result = swapped_trace(swap)
    @test !is_error(result)
    logits = unwrap(result).logits
    @test settled_logits(swap.before, logits, swap.precision, REDRAW_SETTINGS)
    @test newton_correction(swap.after, logits, swap.precision) > 10 * REDRAW_SETTINGS.corrector.logit_tolerance
end

@testset "every endpoint solves the map in effect at the target, whatever the swap level" begin
    rng = fresh_rng()
    for level in [rand(rng, Float32, 8); 1.0f0]
        swap = MapSwap(NEAR_SWAP.before, NEAR_SWAP.after, NEAR_SWAP.precision, level, NEAR_SWAP.width)
        result = swapped_trace(swap)
        @test !is_error(result)
        # The swap lands on the first accepted point at or past `level`; on the endpoint it is not installed.
        stopped = trace_endpoint(LogitGame(swap.before, swap.precision), swap.width, REDRAW_SETTINGS; keep_going = StopAt(level))
        installed = is_error(stopped) && unwrap_error(stopped).context.relative_precision < 1
        in_effect = ifelse(installed, swap.after, swap.before)
        @test on_map(result, in_effect, swap.precision)
    end
end

@testset "a corrector at rounding noise fails PrecisionExhausted at its precision, before the target" begin
    game = LogitGame(GAME, TARGET_PRECISION)
    traced = unwrap(trace_endpoint(game, WIDTH, REDRAW_SETTINGS))
    progress = GameTracing.TraceProgress(copy(traced.logits), 0.5f0, 3, 0.0f0, 0.5f0, zeros(Float32, WIDTH), 0.0f0, false, false)
    exhausted = (; relative_precision = 0.5f0, correction = 2.0f-3)
    outcome = (; step_length = 0.1f0, accepted_step = 0.2f0, exhausted, logit_accuracy = 3.0f-5, unresolved = false, exception = nothing)
    failure = unwrap_error(GameTracing.reached_state(progress, outcome, game, REDRAW_SETTINGS, (;)))
    @test failure.cause == GameTracing.PrecisionExhausted
    @test failure.context.lambda == TARGET_PRECISION * exhausted.relative_precision
    @test failure.context.correction == exhausted.correction
    @test failure.context.logit_accuracy == outcome.logit_accuracy
    @test failure.context.accepted_step == outcome.accepted_step
    unaccepted = GameTracing.TraceProgress(copy(traced.logits), 0.5f0, 0, 0.0f0, 0.5f0, zeros(Float32, WIDTH), 0.0f0, false, false)
    @test iszero(unwrap_error(GameTracing.reached_state(unaccepted, outcome, game, REDRAW_SETTINGS, (;))).context.accepted_step)
    reached = GameTracing.TraceProgress(copy(traced.logits), 1.0f0, 3, 0.0f0, 1.0f0, zeros(Float32, WIDTH), 0.0f0, false, false)
    endpoint = unwrap(GameTracing.reached_state(reached, (; outcome..., exhausted = nothing), game, REDRAW_SETTINGS, (;)))
    @test endpoint.steps == 3
    @test endpoint.logit_accuracy == outcome.logit_accuracy
end

@testset "a redrawn game must keep the trace's target precision, and a hook's ShiftedGame must come from this point" begin
    game = LogitGame(GAME, TARGET_PRECISION)
    slot = GameTracing.RedrawSlot(game, false, 0.1f0)
    check = GameTracing.RootShift(Float32[0.1f0, 0.2f0, 0.5f0], Float32[0, 0, 1], nothing)
    @test GameTracing.swapped(slot, check, nothing) === slot
    @test GameTracing.swapped(slot, check, slot.game) === slot
    @test GameTracing.swapped(slot, check, check(slot.game)) === slot
    @test_throws ArgumentError GameTracing.swapped(slot, check, LogitGame(GAME, 2 * TARGET_PRECISION))
    flipped = check(LogitGame(LinearGainPass(GAME.interaction, -GAME.stand_alone), TARGET_PRECISION))
    next = GameTracing.swapped(slot, check, flipped)
    @test next.pending
    @test next.game === flipped.game
    @test next.factorization === flipped.factorization
    @test next.step_length == slot.step_length
    @test !slot.pending
    elsewhere = GameTracing.RootShift(copy(check.point), check.border, nothing)(game)
    @test_throws ArgumentError GameTracing.swapped(slot, check, elsewhere)
end

@testset "a redraw of random swap 6 at relative precision 0.734 lands on the branch a double-precision reference follows" begin
    (; swap, reference_after) = random_swap(6)
    # A Newton correction moves this swap's logits 0.439 from the old root.
    level = 0.734f0
    stopped = trace_endpoint(LogitGame(swap.before, swap.precision), swap.width, REDRAW_SETTINGS; keep_going = StopAt(level))
    (; logits, relative_precision) = unwrap_error(stopped).context
    @test relative_precision >= level
    result = trace_endpoint(LogitGame(swap.before, swap.precision), swap.width, REDRAW_SETTINGS; redraw = SwapAt(LogitGame(swap.after, swap.precision), level))
    @test !is_error(result)
    endpoint = unwrap(result).logits
    reference = branch_reference(reference_after, swap.width, swap.precision; start = logits .* float(1), start_precision = relative_precision * swap.precision)
    @test reference.reached
    @test reference.crossings == 0
    @test settled_logits(reference_after, endpoint, swap.precision, REDRAW_SETTINGS)
    # The bound reads the endpoint's own double-precision residual.
    residual = 2 * residual_norm(reference_after, endpoint .* float(1), swap.precision * float(1))
    @test maximum(abs, inv.(1 .+ exp.(-endpoint)) .- reference.first_choices) < probability_bound(reference_after, reference.logits, swap.precision, residual, 0.25f0)
end

@testset "a redraw onto an equal map ends where the trace without it ends" begin
    plain = unwrap(trace_endpoint(LogitGame(GAME, TARGET_PRECISION), WIDTH, REDRAW_SETTINGS)).logits
    # An equal map in a new object: the swap restarts the trace from the same root, on the same branch.
    twin = LinearGainPass(copy(GAME.interaction), copy(GAME.stand_alone))
    result = trace_endpoint(LogitGame(GAME, TARGET_PRECISION), WIDTH, REDRAW_SETTINGS; redraw = SwapAt(LogitGame(twin, TARGET_PRECISION), 0.3f0))
    @test !is_error(result)
    @test settled_logits(GAME, unwrap(result).logits, TARGET_PRECISION, REDRAW_SETTINGS)
    @test maximum(abs, unwrap(result).logits .- plain) < 1.0f-3
end

@testset "a redraw onto a game that is non-finite at the redraw fails as a broken solve, at the swap" begin
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
    @test failure.cause in (GameTracing.TraceStuck, GameTracing.Singular)
    @test failure.context.relative_precision == swap_point(NEAR_SWAP).relative_precision
end

@testset "a redraw whose new map's branch meets a pitchfork at the swap fails stuck there, on the old map" begin
    swap = swap_point(PITCHFORK_SWAP)
    level = swap.relative_precision * PITCHFORK_SWAP.precision
    (; before, after) = PITCHFORK_SWAP
    # Between the maps' interactions only the antisymmetric mode changes sign: the new map's root is past a pitchfork.
    modes = [
        symmetric_modes((1 - blend) * before.interaction[1, 2] + blend * after.interaction[1, 2], level)
            for blend in range(0, 1; length = 101)
    ]
    @test all(mode.symmetric > 0 for mode in modes)
    @test first(modes).antisymmetric > 0 > last(modes).antisymmetric
    @test stuck_at_swap(PITCHFORK_SWAP)
end

"""A `redraw` hook that swaps in a fresh copy of `game` at every accepted point: an equal map, a new game each time."""
struct EveryPointCopy{G}
    game::G
end

function (hook::EveryPointCopy{G})(::AbstractVector, ::Real, ::Any)::G where {G}
    return deepcopy(hook.game)
end

@testset "a redraw at the last step the budget allows fails NoConvergence there, with no segment after it" begin
    game = LogitGame(GAME, TARGET_PRECISION)
    result = trace_endpoint(game, WIDTH, TraceSettings{Float32}(; max_steps = 3); redraw = EveryPointCopy(game))
    @test is_error(result)
    @test unwrap_error(result).cause == GameTracing.NoConvergence
    @test unwrap_error(result).context.relative_precision < 1
end
