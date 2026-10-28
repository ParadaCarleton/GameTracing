using ErrorTypes: is_error, unwrap, unwrap_error
using Random: AbstractRNG, Xoshiro

# Random normal-form games whose trace to precision 1e6 once opened at λ ≈ 42426 and stalled off the map.
const FAR_OPENING_SEED = 0xb0bf341028e24ba6
# The seed whose (2, 2, 2) game stalled at λ ≈ 22.9 on its way to 1e6: the corrector floor missed `P G`'s rounding.
const MIXED_STALL_SEED = 0x233ce38d667063a1

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

# Each comparison runs a Gambit trace; the mutation jobs skip it to stay under their time limit.
if get(ENV, "GAMETRACING_MUTATION", "false") != "true"
    @testset "a trace to a far target stays on Gambit's branch" begin
        stalled = last(far_opening_games(Xoshiro(MIXED_STALL_SEED)))
        # The seed still draws the game that stalled: its first payoff's bits as the failing run logged them.
        @test reinterpret(UInt32, first(first(stalled))) == 0x40e91772
        seeded = (far_opening_games(Xoshiro(FAR_OPENING_SEED))..., stalled)
        for payoffs in (seeded..., far_opening_games(fresh_rng())...)
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
end

@testset "the corrector floor covers the rounding of P G, not only the gaps" begin
    gains = NormalFormGains((Float32[7 -2; 3 4], Float32[-2 1; 1 -1]))
    game = LogitGame(gains, FAR_TARGET)
    @test GameTracing.residual_scale(game) == FAR_TARGET * 7
    floor = GameTracing.corrector_tolerance(1.0f-5, 32, Float32[1, 0], GameTracing.residual_scale(game))
    @test floor >= 32 * eps(Float32) * FAR_TARGET * 7
    @test GameTracing.corrector_tolerance(1.0f-5, 32, Float32[1, 0], 0.0f0) == 1.0f-5
end

@testset "the corrector holds the relative precision at its pin, whatever the gap floor" begin
    check = GameTracing.CorrectorReach(
        GameTracing.AcceptAll(),
        GameTracing.CorrectorSettings{Float32}(),
        TraceGeometry{Float32}(),
        zero(Float32),
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

"""A `keep_going` that writes each accepted point of a trace to `target` into `precisions` and `logits`, and stops past `ceiling`."""
struct PointLog{E <: Real}
    target::E
    ceiling::E
    precisions::Vector{E}
    logits::Matrix{E}
end

function (logbook::PointLog)(logits::AbstractVector, relative_precision::Real)::Bool
    point = something(findfirst(isnan, logbook.precisions))
    setindex!(logbook.precisions, relative_precision * logbook.target, point)
    view(logbook.logits, :, point) .= logits
    return relative_precision * logbook.target < logbook.ceiling
end

@testset "the target only caps a log-precision trace: each point to 1e6 up to 100 is the endpoint of a trace to its precision" begin
    gains = NormalFormGains(first(far_opening_games(Xoshiro(FAR_OPENING_SEED))))
    width = GameTracing.logit_count(gains)
    steps = LOG_PRECISION_SETTINGS.max_steps + 1
    logbook = PointLog(FAR_TARGET, 100.0f0, fill(Float32(NaN32), steps), zeros(Float32, width, steps))
    trace_endpoint(LogitGame(gains, FAR_TARGET), width, LOG_PRECISION_SETTINGS; keep_going = logbook)
    recorded = 1:count(!isnan, logbook.precisions)
    for (precision, logits) in zip(view(logbook.precisions, recorded), eachcol(view(logbook.logits, :, recorded)))
        near = trace_endpoint(LogitGame(gains, precision), width, LOG_PRECISION_SETTINGS)
        @test !is_error(near)
        if !is_error(near)
            ours = GameTracing.mixed_strategies(gains, logits)
            gap = maximum(maximum.(abs, ours .- GameTracing.mixed_strategies(gains, unwrap(near).logits)))
            # Each trace's logit residual is under the corrector tolerance at ρ = 1, where gap and logit residuals agree.
            corrector = LOG_PRECISION_SETTINGS.corrector
            residual = GameTracing.corrector_tolerance(corrector.residual_tolerance, corrector.tolerance_floor_ulps, [logits; 1.0f0], GameTracing.residual_scale(LogitGame(gains, precision)))
            @test gap < 2 * probability_bound(gains, logits, precision, residual, 0.5f0)
        end
    end
    @info "accepted points of the trace to 1e6 up to precision 100" points = length(recorded)
end
