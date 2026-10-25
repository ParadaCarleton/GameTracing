using Random: RandomDevice, Xoshiro

"""A random game of `width` logits whose map contracts at `precision`: the equilibrium is unique."""
function contracting_game(rng::Xoshiro, width::Integer, precision::E)::LinearGainPass{
    E,
} where {E <: Real}
    interaction = E.(2 .* rand(rng, width, width) .- 1)
    # The logistic slope is at most 1/4, so a row sum of |A| below 4 / precision makes the map contract.
    scale = 3 / (precision * maximum(sum(abs, interaction; dims = 2)))
    return LinearGainPass(
        interaction .* min(one(E), E(scale)),
        E.(2 .* rand(rng, width) .- 1),
    )
end

@testset "random contracting games: the trace finds the unique equilibrium" begin
    seed = rand(RandomDevice(), UInt)
    rng = Xoshiro(seed)
    for trial in 1:12
        width = rand(rng, 2:4)
        precision = 1 + 3 * rand(rng, Float32)
        pass = contracting_game(rng, width, precision)
        reference = picard_equilibrium(pass, width, precision, 400)
        for settings in (
                TraceSettings{Float32}(),
                TraceSettings{Float32}(tangent = SecantTangent),
                TraceSettings{Float32}(bordered_solver = BorderingSolve),
            )
            traced = trace_endpoint(LogitGame(pass, precision), width, settings)
            @test traced isa Traced
            if traced isa Traced
                @test residual_norm(pass, traced.logits, precision) < 1.0f-3
                @test maximum(abs, traced.logits .- reference) < 1.0f-2
            else
                @info "trace failed" seed trial width precision traced.cause
            end
        end
    end
end
