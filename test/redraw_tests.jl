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
    # The failure keeps the accepted point on the old map, not a root of the new one.
    @test residual_norm(FAR_SWAP.before, failure.context.logits, failure.context.relative_precision * FAR_SWAP.precision) <
        1.0f-3
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
