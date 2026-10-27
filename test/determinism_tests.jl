using ErrorTypes: is_error, unwrap, unwrap_error

"""`payoffs` in Float32."""
function single_precision_payoffs(payoffs::Tuple)::Tuple
    return map(table -> Float32.(table), payoffs)
end

@testset "a trace is deterministic: two traces of one game in one process take the same steps to the same end" begin
    games = [values(LOGITNASH_GAMES)...; OVERSHOT_GAME; UNFOLDED_PITCHFORK_GAME]
    for payoffs in games, (_, settings) in ["default" => TraceSettings{Float32}(); arm_variants()]
        game = LogitGame(NormalFormPass(single_precision_payoffs(payoffs)), 3.0f0)
        first_run = trace_endpoint(game, length(payoffs), settings)
        second_run = trace_endpoint(game, length(payoffs), settings)
        @test is_error(first_run) == is_error(second_run)
        if is_error(first_run) && is_error(second_run)
            @test unwrap_error(first_run).cause == unwrap_error(second_run).cause
            @test unwrap_error(first_run).context.relative_precision == unwrap_error(second_run).context.relative_precision
        elseif !is_error(first_run) && !is_error(second_run)
            @test unwrap(first_run).steps == unwrap(second_run).steps
            @test unwrap(first_run).logits == unwrap(second_run).logits
        end
    end
end
