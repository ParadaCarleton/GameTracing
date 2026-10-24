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
            TraceSettings{Float32}(tangent = SecantTangent),
            TraceSettings{Float32}(bordered_solver = BorderingSolve),
            TraceSettings{Float32}(step_control = nothing),
            TraceSettings{Float32}(step_control = StepControl{Float32}(
                max_distance = nothing,
                max_contraction = nothing,
            )),
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
        TraceSettings{Float32}(max_steps = 1, first_step = 0.005f0, max_step = 0.01f0),
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
    @test residual_norm(
        GAME,
        result.context.logits,
        TARGET_PRECISION * result.context.relative_precision,
    ) <
        1.0f-4
end

"""A `keep_going` callback that goes on only while the point it sees solves the game's map at its precision."""
struct OnTheBranch{P, E <: Real}
    pass::P
    target_precision::E
end

function (probe::OnTheBranch)(logits::AbstractVector, relative_precision::Real)::Bool
    return residual_norm(probe.pass, logits, probe.target_precision * relative_precision) <
        1.0f-4
end

@testset "callbacks see logits" begin
    result = trace_endpoint(
        LogitGame(GAME, TARGET_PRECISION),
        WIDTH,
        TraceSettings{Float32}();
        keep_going = OnTheBranch(GAME, TARGET_PRECISION),
    )
    @test result isa Traced
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
