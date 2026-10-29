using ErrorTypes: is_error, unwrap, unwrap_error
using Random: AbstractRNG, Xoshiro

# Random normal-form games whose trace to precision 1e6 once opened at λ ≈ 42426 and stalled off the map.
const FAR_OPENING_SEED = 0xb0bf341028e24ba6
# The seed whose (2, 2, 2) game stalled at λ ≈ 22.9 on its way to 1e6, under a residual tolerance below P G's rounding.
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
        seeded = (far_opening_games(Xoshiro(FAR_OPENING_SEED))..., last(far_opening_games(Xoshiro(MIXED_STALL_SEED))))
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

@testset "the corrector holds the relative precision at its pin, whatever the logit correction" begin
    check = GameTracing.CorrectorReach(
        GameTracing.AcceptAll(),
        GameTracing.CorrectorSettings{Float32}(),
        TraceGeometry{Float32}(),
        zeros(Float32, 3),
        zeros(Float32, 3),
        zero(Float32),
    )
    point = Float32[1.0f6, -1.0f6, 0.5f0]
    # A correction that moved no logit settles only at the pin, in Newton's `fx` or PALC's `res_f`.
    @test !GameTracing.settled(check, (; x = point, fx = Float32[0, 0, -0.04f0]), 0.0f0)
    @test GameTracing.settled(check, (; x = point, fx = zeros(Float32, 3)), 0.0f0)
    @test !GameTracing.settled(check, (; x = point, res_f = Float32[0, 0, -0.04f0]), 0.0f0)
    tolerance = check.settings.logit_tolerance
    @test GameTracing.settled(check, (; x = point, res_f = zeros(Float32, 3)), tolerance)
    @test !GameTracing.settled(check, (; x = point, res_f = zeros(Float32, 3)), nextfloat(tolerance))
end

"""A fresh `CorrectorReach` whose current step's predictor moved the logits `predictor_move`."""
function fresh_check(predictor_move::E)::GameTracing.CorrectorReach{GameTracing.AcceptAll, E} where {E <: Real}
    check = GameTracing.CorrectorReach(
        GameTracing.AcceptAll(), CorrectorSettings{E}(), TraceGeometry{E}(), zeros(E, 3), zeros(E, 3), one(E),
    )
    check.predictor_move = predictor_move
    return check
end

"""The verdicts of `check` on one solve whose iterates, at relative precision one and at the pin, have first logit `firsts`."""
function solve_verdicts(check::GameTracing.CorrectorReach, firsts::AbstractVector{E})::Vector where {E <: Real}
    return [GameTracing.corrected!(check, (; x = E[first, 0, 1], res_f = zeros(E, 3), step), true) for (step, first) in zip(0:length(firsts), firsts)]
end

@testset "a correction that fails to contract halves the step unless it outgrows the predictor; two such are noise" begin
    # Corrections of 1e-2 then 1e-2: the second is past 0.6 of the first.
    stalled = Float32[0, 1.0f-2, 2.0f-2, 3.0f-2]
    # A predictor that moved the logits by 1 overshot: halving shrinks it, so the corrector stops and nothing is exhausted.
    poor = fresh_check(1.0f0)
    @test solve_verdicts(poor, stalled)[1:3] == [true, true, false]
    @test isnothing(poor.exhausted)
    # It overshoots again at half the step, then the step recovers: the corrections contract and settle.
    poor.predictor_move = 0.5f0
    @test solve_verdicts(poor, stalled[1:3]) == [true, true, false]
    settling = solve_verdicts(poor, Float32[0, 1.0f-2, 1.1f-2, 1.101f-2])
    @test settling[1:3] == [true, true, true]
    @test last(settling) isa GameTracing.BifurcationKit.NewtonConverged
    @test poor.settled_here
    @test isnothing(poor.exhausted)
    # A predictor that moved less than the stalled correction cannot be halved under it: the second stall is noise.
    noisy = fresh_check(1.0f-3)
    @test solve_verdicts(noisy, stalled) == [true, true, true, false]
    @test noisy.exhausted.relative_precision == 1
    @test noisy.exhausted.correction ≈ 1.0f-2
    # Every later solve refuses at its start.
    @test solve_verdicts(noisy, Float32[0]) == [false]
end

@testset "a pure equilibrium traced in Float32 to 1e6 fails PrecisionExhausted where a logit's rounding passes the tolerance" begin
    gains = NormalFormGains(map(table -> Float32.(table), LOGITNASH_GAMES.prisoners_dilemma))
    settings = LOG_PRECISION_SETTINGS
    failure = unwrap_error(trace_endpoint(LogitGame(gains, FAR_TARGET), GameTracing.logit_count(gains), settings))
    @test failure.cause == GameTracing.PrecisionExhausted
    (; lambda, correction, logits) = failure.context
    @test 1 < lambda < FAR_TARGET
    @test correction > settings.corrector.logit_tolerance
    # The logits grow as λ times the payoff gaps; Newton settles to their rounding, an ulp of the largest.
    @test settings.corrector.logit_tolerance / 16 <= eps(maximum(abs, logits)) <= 16 * settings.corrector.logit_tolerance
    @info "precision exhausted" lambda correction largest_logit = maximum(abs, logits)
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
            # The bound reads both points' own double-precision residuals.
            residual = max(residual_norm(gains, logits .* float(1), precision * float(1)), residual_norm(gains, unwrap(near).logits .* float(1), precision * float(1)))
            @test gap < 2 * probability_bound(gains, logits, precision, residual, 0.5f0)
        end
    end
    @info "accepted points of the trace to 1e6 up to precision 100" points = length(recorded)
end
