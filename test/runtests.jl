using GameTracing
using Test

"""Logit response `z = precision (A p + b)` with `p = logistic(z)`: payoff gains `A p + b`, `T` zero at precision zero."""
struct LinearGainPass{E <: Real}
    interaction::Matrix{E}
    stand_alone::Vector{E}
end

function (pass::LinearGainPass)(z::AbstractVector{E}, precision::Real) where {E <: Real}
    stays = inv.(one(E) .+ exp.(-z))
    gains = pass.interaction * stays .+ pass.stand_alone
    slopes = stays .* (one(E) .- stays)
    return (;
        response = precision .* gains,
        logit_slope = precision .* pass.interaction .* transpose(slopes),
        precision_slope = gains,
    )
end

residual_norm(pass::LinearGainPass, logits::AbstractVector, precision::Real) = maximum(
    abs,
    logits .- pass(logits, precision).response,
)

const GAME = LinearGainPass(Float32[0 1; 1 0], Float32[0.5, -0.25])
const TARGET_PRECISION = 2.0f0
const WIDTH = 2

@testset "trace reaches a fixed point of the logit map" begin
    result = trace_endpoint(
        LogitGame(GAME, TARGET_PRECISION),
        WIDTH,
        TraceSettings{Float32}(),
    )
    @test result isa Traced
    @test residual_norm(GAME, result.logits, TARGET_PRECISION) < 1.0f-4
end

@testset "tangent and bordered solver choices reach the same unique equilibrium" begin
    game = LogitGame(GAME, TARGET_PRECISION)
    reference = trace_endpoint(game, WIDTH, TraceSettings{Float32}()).logits
    for settings in (
            TraceSettings{Float32}(tangent = :secant),
            TraceSettings{Float32}(bordered_solver = :bordering),
            TraceSettings{Float32}(step_control = false),
        )
        traced = trace_endpoint(game, WIDTH, settings)
        @test traced isa Traced
        @test maximum(abs, traced.logits .- reference) < 1.0f-3
    end
end

@testset "a spent step budget is a failure value" begin
    result = trace_endpoint(
        LogitGame(GAME, TARGET_PRECISION),
        WIDTH,
        TraceSettings{Float32}(steps = 1, ds = 0.005f0, dsmax = 0.01f0),
    )
    @test result isa TraceFailure
    @test result.cause in (GameTracing.NoConvergence, GameTracing.TraceStuck)
end

@testset "a refusing keep_going callback stops the trace" begin
    refuse_all(::AbstractVector, ::Real) = false
    result = trace_endpoint(
        LogitGame(GAME, TARGET_PRECISION),
        WIDTH,
        TraceSettings{Float32}();
        keep_going = refuse_all,
    )
    @test result.cause == GameTracing.TraceStuck
    @test result.context.relative_precision < 1
end

@testset "a non-finite game fails at the start" begin
    broken = LinearGainPass(Float32[NaN 1; 1 0], Float32[0.5, -0.25])
    result = trace_endpoint(
        LogitGame(broken, TARGET_PRECISION),
        WIDTH,
        TraceSettings{Float32}();
        context = (; label = :broken),
    )
    @test result.cause == GameTracing.NotFinite
    @test result.context.label == :broken
end

@testset "augmented_theta rejects a trace PALC cannot weight" begin
    @test GameTracing.augmented_theta(0.5f0, 3) ≈ 0.5f0 * 4 / 3
    @test_throws ArgumentError GameTracing.augmented_theta(0.9f0, 3)
end
