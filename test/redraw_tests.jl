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
    # Probabilities to 1e-3, as the LogitNash cross-check compares a Float32 trace with its reference.
    @test maximum(abs, inv.(1 .+ exp.(-logits)) .- reference.first_choices) < 1.0f-3
    # The swap moved the branch: the endpoint is not the old map's.
    @test residual_norm(NEAR_SWAP.before, logits, NEAR_SWAP.precision) > 100 * tolerance
    @info "bytes per redraw" swapped = @allocated(swapped_trace(NEAR_SWAP)) unswapped = @allocated(
        trace_endpoint(LogitGame(NEAR_SWAP.before, NEAR_SWAP.precision), NEAR_SWAP.width, REDRAW_SETTINGS)
    )
end

@testset "a redraw whose new root is past the correction's reach fails stuck and projected" begin
    result = swapped_trace(FAR_SWAP)
    @test is_error(result)
    failure = unwrap_error(result)
    @test failure.cause == GameTracing.TraceStuck
    @test failure.context.projected
    @test failure.context.relative_precision >= FAR_SWAP.swap_level
    @test failure.context.precision == failure.context.relative_precision * FAR_SWAP.precision
    @test failure.context.step_length > 0
    # The failure keeps the accepted point on the old map, not a root of the new one.
    @test residual_norm(FAR_SWAP.before, failure.context.logits, failure.context.relative_precision * FAR_SWAP.precision) <
        1.0f-3
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
    slot = GameTracing.RedrawSlot(LogitGame(GAME, TARGET_PRECISION), false, 0.1f0)
    @test !GameTracing.swapped!(slot, nothing)
    @test !GameTracing.swapped!(slot, slot.game)
    @test !slot.pending
    @test_throws ArgumentError GameTracing.swapped!(slot, LogitGame(GAME, 2 * TARGET_PRECISION))
    @test GameTracing.swapped!(slot, LogitGame(LinearGainPass(GAME.interaction, -GAME.stand_alone), TARGET_PRECISION))
    @test slot.pending
end
