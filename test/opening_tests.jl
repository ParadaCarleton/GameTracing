using ErrorTypes: unwrap_error
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

"""A `keep_going` that writes each accepted precision of a trace to `target` into `precisions` and stops past `ceiling`."""
struct PrecisionLog{E <: Real}
    target::E
    ceiling::E
    precisions::Vector{E}
end

function (logbook::PrecisionLog)(::AbstractVector, relative_precision::Real)::Bool
    setindex!(logbook.precisions, relative_precision * logbook.target, something(findfirst(isnan, logbook.precisions)))
    return relative_precision * logbook.target < logbook.ceiling
end

"""The accepted precisions of a trace of `gains` to `target` in `settings`, up to the first past `ceiling`."""
function accepted_precisions(gains::NormalFormGains, target::E, ceiling::E, settings::TraceSettings{E})::AbstractVector{E} where {E <: Real}
    logbook = PrecisionLog(target, ceiling, fill(E(NaN32), settings.max_steps + 1))
    trace_endpoint(LogitGame(gains, target), GameTracing.logit_count(gains), settings; keep_going = logbook)
    return filter(!isnan, logbook.precisions)
end

@testset "the target only caps a trace: to 100 and to 1e6 it accepts the same precisions up to 100, in $distance" for distance in instances(StepDistance)
    settings = TraceSettings{Float32}(; geometry = TraceGeometry{Float32}(; distance))
    gains = NormalFormGains(first(far_opening_games(Xoshiro(FAR_OPENING_SEED))))
    near, far = (accepted_precisions(gains, target, 100.0f0, settings) for target in (100.0f0, FAR_TARGET))
    shared = min(length(near), length(far))
    # Two traces of one branch agree to the corrector's tolerance on each step's precision.
    agree = length(near) == length(far) && all(abs.(view(near, 1:shared) .- view(far, 1:shared)) .<= 1.0f-3 .* (1 .+ view(near, 1:shared)))
    @test agree
    if !agree
        @error "the target moved the accepted precisions" distance near = repr(first(near, 16)) far = repr(first(far, 16)) lengths = (length(near), length(far))
    end
end
