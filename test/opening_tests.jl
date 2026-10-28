using ErrorTypes: unwrap_error
using Logging: ConsoleLogger, Debug, with_logger
using Random: AbstractRNG, Xoshiro

# Random normal-form games whose trace to precision 1e6 once opened at λ ≈ 42426 and stalled off the map.
const FAR_OPENING_SEED = 0xb0bf341028e24ba6

"""Normal-form games with strategy counts `(3, 3)` and `(2, 2, 2)`, payoffs `3 randn`, drawn from `rng` in that order."""
function far_opening_games(rng::AbstractRNG)::AbstractVector
    return [ntuple(_ -> 3 .* randn(rng, Float32, counts), length(counts)) for counts in ((3, 3), (2, 2, 2))]
end

"""A `keep_going` that stops a trace to `target` at the first accepted point at or past precision `precision`."""
struct StopAtPrecision{E <: Real}
    precision::E
    target::E
end

(stop::StopAtPrecision)(::AbstractVector, relative_precision::Real)::Bool = relative_precision * stop.target < stop.precision

const FAR_TARGET = 1.0f6
const LOG_PRECISION_SETTINGS = TraceSettings{Float32}(; geometry = TraceGeometry{Float32}(; distance = FisherLogPrecisionDistance))

# SCRATCH: every corrector iterate and point of one far-target opening, to find where its first step is set.
let gains = NormalFormGains(first(far_opening_games(Xoshiro(FAR_OPENING_SEED))))
    with_logger(ConsoleLogger(stderr, Debug)) do
        trace_endpoint(
            LogitGame(gains, FAR_TARGET),
            GameTracing.logit_count(gains),
            LOG_PRECISION_SETTINGS;
            keep_going = StopAtPrecision(zero(Float32), FAR_TARGET),
        )
    end
end

@testset "a trace to a far target opens one first step from the flat start in its own distance" begin
    for payoffs in (far_opening_games(Xoshiro(FAR_OPENING_SEED))..., far_opening_games(fresh_rng())...)
        gains = NormalFormGains(payoffs)
        failure = unwrap_error(trace_endpoint(
            LogitGame(gains, FAR_TARGET),
            GameTracing.logit_count(gains),
            LOG_PRECISION_SETTINGS;
            keep_going = StopAtPrecision(zero(Float32), FAR_TARGET),
        ))
        # A first step of length `first_step` in `log(1 + λ)` reaches at most `expm1(first_step)`.
        @test 0 < failure.context.precision <= 2 * expm1(LOG_PRECISION_SETTINGS.first_step)
    end
end

@testset "a trace to a far target stays on Gambit's branch" begin
    for payoffs in (far_opening_games(Xoshiro(FAR_OPENING_SEED))..., far_opening_games(fresh_rng())...)
        gains = NormalFormGains(payoffs)
        failure = unwrap_error(trace_endpoint(
            LogitGame(gains, FAR_TARGET),
            GameTracing.logit_count(gains),
            LOG_PRECISION_SETTINGS;
            keep_going = StopAtPrecision(50.0f0, FAR_TARGET),
        ))
        (; precision, logits) = failure.context
        @test precision >= 50
        if precision < 50
            @error "a far-target trace stopped short" failure.cause failure.context
        end
        oracle = gambit_strategies(payoffs, precision)
        gap = maximum(maximum.(abs, GameTracing.mixed_strategies(gains, logits) .- oracle.strategies))
        @test gap < 1.0f-3
    end
end

@testset "the corrector holds the relative precision at its pin, whatever the gap floor" begin
    check = GameTracing.CorrectorReach(
        GameTracing.AcceptAll(),
        GameTracing.CorrectorSettings{Float32}(),
        TraceGeometry{Float32}(),
        zeros(Float32, 3),
        zeros(Float32, 3),
        zero(Float32),
    )
    # Gaps of 1e6 put the gap floor near 4, far above a relative precision 0.04 from its pin.
    point = Float32[1.0f6, -1.0f6, 0]
    @test GameTracing.settled_tolerance(check, point) > 0.04f0
    @test !GameTracing.settled(check, (; x = point, fx = Float32[0, 0, -0.04f0], residual = 0.04f0))
    @test GameTracing.settled(check, (; x = point, fx = zeros(Float32, 3), residual = zero(Float32)))
    @test !GameTracing.settled(check, (; x = point, res_f = Float32[0, 0, -0.04f0], residual = 0.04f0))
end
