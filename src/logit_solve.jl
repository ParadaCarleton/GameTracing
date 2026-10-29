using ErrorTypes: Err, Ok, Result, is_error, unwrap, unwrap_error
using LinearAlgebra: dot

"""
    LogitQREProfile(gains, lam, logits)

One point of a normal-form game's logit QRE branch, as pygambit's `LogitQREMixedStrategyProfile`: the precision `lam`
(Gambit's `λ`, `π ∝ exp(λ u)`, the trace's precision) and `profile`, each player's mixed strategy (`mixed_strategies`).
`logits` are the trace's log odds against each player's first strategy; they stay finite where a probability underflows.
"""
struct LogitQREProfile{E <: Real, S <: Tuple, L <: AbstractVector{E}}
    lam::E
    profile::S
    logits::L
    # The only constructor: `profile` is computed from `logits`, so the two cannot disagree.
    function LogitQREProfile(gains::NormalFormGains, lam::E, logits::AbstractVector{E})::LogitQREProfile{E} where {E <: Real}
        stored = collect(logits)
        profile = mixed_strategies(gains, stored)
        return new{E, typeof(profile), typeof(stored)}(lam, profile, stored)
    end
end

# The results of `logit_solve` and of the branch searches, over `N` players in element type `E`.
const QREProfile{E, N} = LogitQREProfile{E, NTuple{N, Vector{E}}, Vector{E}}
const ProfileResult{E, N} = Result{QREProfile{E, N}, TraceFailure}
const BranchResult{E, N} = Result{Vector{QREProfile{E, N}}, TraceFailure}

"""
    gambit_settings(first_step, max_accel)

`TraceSettings` for pygambit's step arguments: `first_step` is the first arclength step and `max_accel` caps each
step's growth (`aggressiveness = max_accel - 1`). The trace steps in `FisherLogPrecisionDistance`, whose precision
term is the step in `log(1 + λ)`, so a trace to a far precision opens at the scale of a near one.
"""
function gambit_settings(first_step::E, max_accel::Real)::TraceSettings{E} where {E <: Real}
    if !(first_step > 0)
        throw(ArgumentError("first_step must be positive; got $first_step"))
    end
    if !(max_accel >= 1)
        throw(ArgumentError("max_accel must be at least 1; got $max_accel"))
    end
    return TraceSettings{E}(;
        first_step,
        aggressiveness = E(max_accel) - one(E),
        geometry = TraceGeometry{E}(; distance = FisherLogPrecisionDistance),
    )
end

"""The regret of a player whose gains against its first strategy are `gain` and whose mixed strategy is `strategy`."""
function player_regret(gain::AbstractVector{E}, strategy::AbstractVector{E})::E where {E <: Real}
    return maximum(gain; init = zero(E)) - dot(view(strategy, (firstindex(strategy) + 1):lastindex(strategy)), gain)
end

"""The largest regret of any player of `gains` at `logits`: its best strategy's expected payoff less its mixed strategy's."""
function max_regret(gains::NormalFormGains, logits::AbstractVector{E})::E where {E <: Real}
    width = length(logits)
    gain = zeros(E, width)
    gains(gain, zeros(E, width, width), zeros(E, width), logits, one(E))
    return maximum(player_regret.(view.((gain,), logit_ranges(gains)), mixed_strategies(gains, logits)))
end

"""Refuse `settings` whose element type is not `float` of the payoffs' of `gains`: the trace and its profiles run in the settings' type."""
function require_payoff_type(gains::NormalFormGains, ::TraceSettings{E})::Nothing where {E <: Real}
    if E != eltype(gains)
        throw(ArgumentError("settings trace in $E, but the payoffs are $(eltype(gains)); build the settings in $(eltype(gains))"))
    end
    return nothing
end

"""The largest payoff of `gains` less its smallest: Gambit's scale for `maxregret`."""
function payoff_range(gains::NormalFormGains{N, A})::eltype(A) where {N, A}
    return maximum(maximum, gains.payoffs) - minimum(minimum, gains.payoffs)
end

"""
    BranchRecorder(gains, target, tolerance, precisions, logits)

A `keep_going` that writes each accepted point of a trace of `gains` to precision `target` into the next branch point
of `precisions` (NaN until written) and `logits`, and stops the trace at the first point whose largest regret
(`max_regret`) is at most `tolerance`. Its arrays hold one branch point per step the trace may take.
"""
struct BranchRecorder{G <: NormalFormGains, E <: Real}
    gains::G
    target::E
    tolerance::E
    precisions::Vector{E}
    logits::Matrix{E}
end

function (recorder::BranchRecorder{G, E})(logits::AbstractVector{E}, relative_precision::Real)::Bool where {G, E <: Real}
    point = something(findfirst(isnan, recorder.precisions))
    setindex!(recorder.precisions, relative_precision * recorder.target, point)
    view(recorder.logits, :, point) .= logits
    return max_regret(recorder.gains, logits) > recorder.tolerance
end

"""The branch points `recorder` wrote: the leading entries of its precisions that are not NaN."""
function recorded_points(recorder::BranchRecorder)::AbstractUnitRange
    return 1:count(!isnan, recorder.precisions)
end

"""The `LogitQREProfile` at precision `lam` on the principal branch of `gains`, traced with `settings`."""
function lambda_profile(gains::NormalFormGains{N}, lam::E, settings::TraceSettings{E})::ProfileResult{E, N} where {N, E <: Real}
    result = trace_endpoint(LogitGame(gains, lam), logit_count(gains), settings; context = (; lam))
    if is_error(result)
        return Err(unwrap_error(result))
    end
    return Ok(LogitQREProfile(gains, lam, unwrap(result).logits))
end

"""
    logit_solve_lambda(gains, lam; first_step, max_accel, settings)

pygambit's `logit_solve_lambda`: the `LogitQREProfile` on the principal branch of the normal-form game `gains`
(`NormalFormGains`) at each precision in `lam`, a number or a vector, in `lam`'s order, or the first trace's failure.
`first_step` and `max_accel` set the trace's steps (`gambit_settings`); `settings` replaces them whole. Profiles take
the payoffs' element type, and `settings` must share it (`ArgumentError` otherwise).
"""
function logit_solve_lambda(
    gains::NormalFormGains{N},
    lam::AbstractVector{<:Real};
    first_step::Real = 0.03f0,
    max_accel::Real = 1.1f0,
    settings::TraceSettings{E} = gambit_settings(float(eltype(gains))(first_step), max_accel),
)::BranchResult{E, N} where {N, E <: Real}
    require_payoff_type(gains, settings)
    # ponytail: one trace per precision; one trace stopping at each would save the repeated low-precision steps.
    results = lambda_profile.((gains,), E.(lam), (settings,))
    for result in results
        if is_error(result)
            return Err(unwrap_error(result))
        end
    end
    return Ok(unwrap.(results))
end

function logit_solve_lambda(
    gains::NormalFormGains{N},
    lam::Real;
    first_step::Real = 0.03f0,
    max_accel::Real = 1.1f0,
    settings::TraceSettings{E} = gambit_settings(float(eltype(gains))(first_step), max_accel),
)::BranchResult{E, N} where {N, E <: Real}
    return logit_solve_lambda(gains, [lam]; settings)
end

"""
    logit_solve_branch(gains; maxregret, first_step, max_accel, maximum_precision, settings)

pygambit's `logit_solve_branch`: every accepted point of the principal branch of the normal-form game `gains`, from
the first step to the first point where no player's regret passes `maxregret` times the payoff range. The trace runs
to `maximum_precision` at most, and fails `NoConvergence` with reason `:maximum_precision` if it ends there unsettled;
any other failure of the trace is returned as is. `maxregret` defaults to `sqrt(eps(E))` in the trace's element type `E`. `first_step`, `max_accel` and `settings` are `logit_solve_lambda`'s.
"""
function logit_solve_branch(
    gains::NormalFormGains{N};
    first_step::Real = 0.03f0,
    max_accel::Real = 1.1f0,
    maximum_precision::Real = 1.0f6,
    settings::TraceSettings{E} = gambit_settings(float(eltype(gains))(first_step), max_accel),
    maxregret::Real = sqrt(eps(typeof(settings.first_step))),
)::BranchResult{E, N} where {N, E <: Real}
    if !(maxregret > 0)
        throw(ArgumentError("maxregret must be positive; got $maxregret"))
    end
    require_payoff_type(gains, settings)
    width = logit_count(gains)
    recorder = BranchRecorder(
        gains,
        E(maximum_precision),
        E(maxregret) * E(payoff_range(gains)),
        fill(E(NaN32), settings.max_steps + 1),
        zeros(E, width, settings.max_steps + 1),
    )
    result = trace_endpoint(LogitGame(gains, E(maximum_precision)), width, settings; keep_going = recorder, context = (; maxregret))
    recorded = recorded_points(recorder)
    # fatou-ignore call-arity: Base `view` takes any number of indices (jolars/fatou#125)
    logits = eachcol(view(recorder.logits, :, recorded))
    settled = !isempty(recorded) && max_regret(gains, last(logits)) <= recorder.tolerance
    if !settled && is_error(result)
        return Err(unwrap_error(result))
    end
    if !settled
        return Err(TraceFailure(NoConvergence, (; maxregret, maximum_precision, reason = :maximum_precision)))
    end
    return Ok(LogitQREProfile.((gains,), view(recorder.precisions, recorded), logits))
end

"""
    logit_solve(gains; maxregret, first_step, max_accel, maximum_precision, settings)

pygambit's `logit_solve`: the last point of `logit_solve_branch`, an approximate Nash equilibrium of the normal-form
game `gains` at which no player's regret passes `maxregret` times the payoff range.
"""
function logit_solve(
    gains::NormalFormGains{N};
    first_step::Real = 0.03f0,
    max_accel::Real = 1.1f0,
    maximum_precision::Real = 1.0f6,
    settings::TraceSettings{E} = gambit_settings(float(eltype(gains))(first_step), max_accel),
    maxregret::Real = sqrt(eps(typeof(settings.first_step))),
)::ProfileResult{E, N} where {N, E <: Real}
    branch = logit_solve_branch(gains; maxregret, maximum_precision, settings)
    if is_error(branch)
        return Err(unwrap_error(branch))
    end
    return Ok(last(unwrap(branch)))
end
