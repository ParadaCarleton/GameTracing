using BifurcationKit: BifurcationKit
using Logging: @debug
"""
    CorrectorSettings{E}(; reach, max_kl, max_angle, logit_tolerance, pin_tolerance, contraction_limit,
                         contraction_offset, newton_iterations)

The corrector's controls (`CorrectorReach`); `nothing` switches a limit off. `reach`: a corrector that moves a logit
further from the predictor than this times the predictor's own largest logit move is refused. `max_kl`: a corrected point whose largest per-logit probability KL from
the predictor passes this is refused. `max_angle`: so is one whose chord turns past this many radians from the
predictor's direction.

Convergence is scale-free, as in Gambit's tracer (path.cc:372, `c_tol` in log probabilities): a Newton correction that
moves no logit more than `logit_tolerance`, at a relative precision within the relative `pin_tolerance` of its pin.
A correction that moves the logits more than `contraction_limit` times the one before it (plus `logit_tolerance`
times `contraction_offset`) fails to contract; BifurcationKit's step growth reads the same rate (`corrector_quality`).
`newton_iterations`: Newton iterations per corrector.
"""
Base.@kwdef struct CorrectorSettings{E <: Real}
    reach::Union{Nothing, E} = 2.0f0
    max_kl::Union{Nothing, E} = nothing
    max_angle::Union{Nothing, E} = nothing
    logit_tolerance::E = 1.0f-4
    pin_tolerance::E = 1.0f-5
    contraction_limit::Union{Nothing, E} = 0.6f0
    contraction_offset::E = 0.1f0
    newton_iterations::Int = 10
end

"""The largest change of a logit from `from` to `to`, two points `[w; ρ]` (`logits_at`)."""
function logit_change(from::AbstractVector{E}, to::AbstractVector{E})::E where {E <: Real}
    return maximum(abs, logits_at(to) .- logits_at(from))
end

"""A trace callback that always lets the trace go on: `(logits, relative_precision)` or `(logits, relative_precision, converged)`."""
struct AcceptAll end

(::AcceptAll)(::AbstractVector, ::Real)::Bool = true
(::AcceptAll)(::AbstractVector, ::Real, ::Bool)::Bool = true

"""Whether `measured` is within `limit`; every measure is within an absent (`nothing`) limit."""
within_limit(::Nothing, ::Real)::Bool = true
within_limit(limit::Real, measured::Real)::Bool = measured <= limit

"""
    NoiseRecord(accuracy, suspect)

A corrector's record over its trace. `accuracy`: the largest last correction of a closing point the corrector
accepted. `suspect`: the last suspected stall at rounding noise, or `nothing`; `confirmed` means it exhausted the
precision.
"""
struct NoiseRecord{E <: Real}
    accuracy::E
    suspect::Union{Nothing, @NamedTuple{relative_precision::E, correction::E, confirmed::Bool}}
end

"""
    CorrectorReach(accept_corrected, settings, geometry, previous, predicted, step_length)

The trace's BifurcationKit `callback_newton`, which also decides the corrector's convergence: a `settled` iterate
ends Newton's method converged (`NewtonConverged`), and a correction that fails to contract stops it (`corrected!`).
Two such corrections in a row, each larger than the predictor's own move in logits, are suspected rounding noise
(`noise`), confirmed at the next halved step; the check then refuses every corrector (`noise_confirmed`).
A non-finite residual stops the corrector,
and so does any iterate whose logits are further from the predictor's than `settings.reach` times the predictor's own
largest logit move (`within_reach`); the step then halves. On a smooth branch that distance is the branch's curvature
times `ds² / 2`, so halving brings any step under it, while an iterate that lands on another branch is far from it. A converged closing point is refused the same way past `reach`, past a largest probability KL
`KL(corrected ‖ predicted)` of `settings.max_kl` (`largest_kl`), or past a turn of `settings.max_angle` radians from
the predictor's logit move (`chord_angle`, from `previous`). A predictor that did not move has no reach to compare.
`accept_corrected(logits, ρ, converged)` then sees every other closing point. `previous`, `predicted` and
`step_length` hold the current step's start and predictor, `predictor_move` the predictor's largest logit change.
`iterate`, `correction`, `stalls` and `settled_here` follow the current solve: its last iterate, its last correction
in logits, its non-contracting corrections in a row, and whether it settled. `noise` (`NoiseRecord`) spans the trace.
"""
mutable struct CorrectorReach{C, E <: Real}
    # Mutable because BifurcationKit shows the predictor only to a corrector's first callback and the
    # accept_corrected point only to its last, and returns neither.
    accept_corrected::C
    settings::CorrectorSettings{E}
    geometry::TraceGeometry{E}
    previous::Vector{E}
    predicted::Vector{E}
    step_length::E
    predictor_move::E
    iterate::Vector{E}
    correction::E
    stalls::Int
    settled_here::Bool
    noise::NoiseRecord{E}
end

"""Whether `check`'s corrector met rounding noise, confirmed at a halved step (`corrected!`)."""
function noise_confirmed(check::CorrectorReach)::Bool
    return !isnothing(check.noise.suspect) && check.noise.suspect.confirmed
end

function CorrectorReach(
    accept_corrected::C,
    settings::CorrectorSettings{E},
    geometry::TraceGeometry{E},
    previous::AbstractVector{E},
    predicted::AbstractVector{E},
    step_length::E,
)::CorrectorReach{C, E} where {C, E <: Real}
    # A plain Newton solve has no predictor move: NaN fails the noise test, so its stalls refuse.
    return CorrectorReach{C, E}(
        accept_corrected, settings, geometry, collect(previous), collect(predicted), step_length,
        E(NaN32), collect(previous), E(NaN32), 0, false, NoiseRecord(zero(E), nothing),
    )
end

"""Whether `check` accepts its corrector's converged closing point `corrected` (reach, KL and angle)."""
function closing_accepted(check::CorrectorReach{C, E}, corrected::AbstractVector{E})::Bool where {C, E <: Real}
    (; settings) = check
    if !within_reach(check, corrected)
        @debug "corrector refused" _group = :trace reason = :reach predictor_move = check.predictor_move
        return false
    end
    if !within_limit(settings.max_kl, largest_kl(corrected, check.predicted))
        @debug "corrector refused" _group = :trace reason = :kl
        return false
    end
    if !within_limit(settings.max_angle, chord_angle(check.previous, check.predicted, corrected))
        @debug "corrector refused" _group = :trace reason = :angle
        return false
    end
    return true
end

"""The verdict that ends a solve: converged (`NewtonConverged`) when it `settled_here`, else `proceed`."""
function closing_verdict(check::CorrectorReach, proceed::Bool)::Union{Bool, BifurcationKit.NewtonConverged}
    if check.settled_here
        return BifurcationKit.NewtonConverged()
    end
    return proceed
end

function (check::CorrectorReach{C, E})(iterate::I;
    fromNewton::Bool = true,
    # fatou-ignore unused-argument: BifurcationKit passes keyword arguments this callback ignores
    keywords...,
)::Union{Bool, BifurcationKit.NewtonConverged} where {C, E <: Real, I <: NamedTuple}
    if fromNewton
        # Newton's closing call is its only one without `J`; an exact root converges on BifurcationKit's own test.
        if !haskey(iterate, :J)
            return closing_verdict(check, isfinite(iterate.residual))
        end
        if iterate.step == 0
            check.predictor_move = E(NaN32)
        end
        return corrected!(check, iterate, isfinite(iterate.residual))
    end
    # PALC's closing call, the only one without `z0`, ends every corrector.
    if haskey(iterate, :z0)
        @debug "corrector iterate" _group = :trace step = iterate.step residual = iterate.residual relative_precision = last(
            iterate.x,
        )
        if iterate.step == 0
            copyto!(check.predicted, iterate.x)
            copyto!(check.previous, iterate.z0.u)
            check.step_length = arclength_distance(iterate.z0.u, iterate.x, check.geometry)
            check.predictor_move = logit_change(iterate.z0.u, iterate.x)
        end
        if !isfinite(iterate.residual)
            @debug "corrector stopped" _group = :trace reason = :nonfinite
            return corrected!(check, iterate, false)
        end
        reached = within_reach(check, iterate.x)
        if !reached
            @debug "corrector stopped" _group = :trace reason = :reach
        end
        return corrected!(check, iterate, reached)
    end
    # An exact root converges unsettled on BifurcationKit's own test (`residual <= tol`, tol 0), so it meets the same tests.
    converged = isfinite(iterate.residual) && (check.settled_here || iszero(iterate.residual))
    @debug "corrector closed" _group = :trace step = iterate.step residual = iterate.residual converged step_length = check.step_length
    if converged && !closing_accepted(check, iterate.x)
        return false
    end
    if !check.accept_corrected(logits_at(iterate.x), iterate.p, converged)
        return false
    end
    # A solve that settled at its start has no correction (NaN).
    if converged && !isnan(check.correction)
        check.noise = NoiseRecord(max(check.noise.accuracy, check.correction), check.noise.suspect)
    end
    return closing_verdict(check, true)
end

"""
    corrected!(check, iterate, proceed)

`check`'s verdict on a Newton `iterate` that its other tests let `proceed`, in logits `z = ρ w`. The first iterate
is the solve's start and never settles. Each later one is a correction `Δz` from the iterate before: it settles
(`NewtonConverged`) when `settled`. A correction past `contraction_limit` times the one before (plus
`logit_tolerance · contraction_offset`) fails to contract, Gambit's test (path.cc), and stops the corrector, so the
step halves. Two in a row, each larger than the predictor's own move, are suspected noise (`check.noise`); the
step still halves. A second such solve before any settles, its correction at least half the suspect's, confirms it
(`noise_confirmed`): halving shrinks the corrections of a predictor past a fold, not noise. Every later solve refuses.
"""
function corrected!(
    check::CorrectorReach{C, E},
    iterate::NamedTuple,
    proceed::Bool,
)::Union{Bool, BifurcationKit.NewtonConverged} where {C, E <: Real}
    if noise_confirmed(check)
        return false
    end
    if iterate.step == 0
        copyto!(check.iterate, iterate.x)
        check.correction, check.stalls, check.settled_here = E(NaN32), 0, false
        return proceed
    end
    correction, before = logit_change(check.iterate, iterate.x), check.correction
    copyto!(check.iterate, iterate.x)
    check.correction = correction
    if !proceed
        return false
    end
    if settled(check, iterate, correction)
        check.settled_here, check.noise = true, NoiseRecord(check.noise.accuracy, nothing)
        return BifurcationKit.NewtonConverged()
    end
    (; contraction_limit, contraction_offset, logit_tolerance) = check.settings
    # The first correction has no rate (`before` is NaN), so it contracts.
    if isnan(before) || within_limit(scaled_limit(contraction_limit, before + logit_tolerance * contraction_offset), correction)
        check.stalls = 0
        return true
    end
    if !(check.predictor_move <= correction)
        @debug "corrector stopped" _group = :trace reason = :contraction correction before
        return false
    end
    check.stalls += 1
    if check.stalls < 2
        return true
    end
    # Halving leaves noise's corrections alike (ratio 1) and shrinks an overshot predictor's by 1/4; 1/2 splits them.
    suspect = check.noise.suspect
    confirmed = !isnothing(suspect) && correction >= suspect.correction / 2
    @debug "corrector stopped" _group = :trace reason = :noise confirmed correction predictor_move = check.predictor_move
    check.noise = NoiseRecord(check.noise.accuracy, (; relative_precision = last(iterate.x), correction, confirmed))
    return false
end

"""The pin row `ρ - pin` of a corrector `iterate`'s residual: the last entry of Newton's `fx` or of PALC's `res_f`."""
function pin_residual(iterate::NamedTuple)::eltype(iterate.x)
    if haskey(iterate, :fx)
        return last(iterate.fx)
    end
    return last(iterate.res_f)
end

"""
Whether `check`'s corrector has settled `iterate`, reached by a Newton `correction` in logits: the correction moved no
logit more than `logit_tolerance`, and the relative precision `ρ` is at its pin to the relative `pin_tolerance`. The pin
needs its own test: a far target's logits barely move with `ρ`, so they would pass a `ρ` that never left its start.
"""
function settled(check::CorrectorReach, iterate::I, correction::Real)::Bool where {I <: NamedTuple}
    relative_precision = last(iterate.x)
    offset = pin_residual(iterate)
    pin_scale = max(abs(relative_precision), abs(relative_precision - offset))
    return correction <= check.settings.logit_tolerance && abs(offset) <= check.settings.pin_tolerance * pin_scale
end

"""
Whether `point` moved no logit further from the predictor's than `check.settings.reach` times the predictor's own
largest logit move (`logit_change`). Logits are target-free, as the convergence test is; a weighted norm can shrink a
branch jump (frozen Fisher weights) or swell a settled point (gaps `w = P G` at a far target).
"""
function within_reach(check::CorrectorReach, point::AbstractVector)::Bool
    if !(check.predictor_move > 0)
        return true
    end
    return within_limit(
        scaled_limit(check.settings.reach, check.predictor_move),
        logit_change(check.predicted, point),
    )
end

"""
    TraceProgress(gaps, relative_precision, steps, precision_min, precision_max, gap_tangent, tangent_precision, unmoved)

The last accepted point `(gaps, relative_precision)` of a trace, its step count, the range of the relative precision
over accepted points and the tangent there (both in payoff gaps): BifurcationKit's `finalise_solution`
callback. An accepted point bitwise equal to the one before it (a step below the point's rounding) sets `unmoved` and
stops the trace, a stall. `JULIA_DEBUG=trace` lists every point.
"""
mutable struct TraceProgress{E <: Real}
    gaps::Vector{E}
    relative_precision::E
    steps::Int
    precision_min::E
    precision_max::E
    gap_tangent::Vector{E}
    tangent_precision::E
    unmoved::Bool
end

function (progress::TraceProgress)(point::P, tangent::T, step::S, ::Any;
    # fatou-ignore unused-argument: BifurcationKit passes keyword arguments this callback ignores
    keywords...,
)::Bool where {P, T, S <: Integer}
    @debug "trace point" _group = :trace step relative_precision = point.p state = copy(
        point.u,
    ) tangent = copy(tangent.u)
    progress.unmoved = point.p == progress.relative_precision &&
        view(point.u, eachindex(progress.gaps)) == progress.gaps
    copyto!(progress.gaps, 1, point.u, 1, length(progress.gaps))
    progress.relative_precision = point.p
    progress.steps = step
    progress.precision_min = min(progress.precision_min, point.p)
    progress.precision_max = max(progress.precision_max, point.p)
    copyto!(progress.gap_tangent, 1, tangent.u, 1, length(progress.gap_tangent))
    progress.tangent_precision = tangent.p
    return !progress.unmoved
end

"""
    TraceFinalise(progress, keep_going, redraw, slot, step_offset)

BifurcationKit's `finalise_solution` of one trace segment: `progress` records the point (its step counted from
`step_offset`, the steps of earlier segments), and the trace goes on while `keep_going(logits, relative_precision)`
holds. Then `redraw(logits, relative_precision)` may return a new game: `slot` takes it and the segment stops, for
`trace_endpoint` to correct the point onto it and restart. A `redraw` that returns `nothing` or the current game swaps nothing.
"""
struct TraceFinalise{R <: TraceProgress, A, D, S}
    progress::R
    keep_going::A
    redraw::D
    slot::S
    step_offset::Int
end

function (finalise::TraceFinalise)(point::P, tangent::T, step::S, result::Any;
    state::Any,
    # fatou-ignore unused-argument: BifurcationKit passes keyword arguments this callback ignores
    keywords...,
)::Bool where {P, T, S <: Integer}
    recorded = finalise.progress(point, tangent, finalise.step_offset + step, result)
    logits = logits_at(point.u)
    if !(recorded && finalise.keep_going(logits, point.p))
        return false
    end
    finalise.slot.step_length = abs(state.ds)
    return !swapped!(finalise.slot, finalise.redraw(logits, point.p))
end

"""One scalar per step for BifurcationKit's branch record, the relative_precision: `TraceProgress` keeps the point."""
function trace_record_precision(::Any, relative_precision::L;
    # fatou-ignore unused-argument: BifurcationKit passes keyword arguments this callback ignores
    keywords...,
)::L where {L <: Real}
    return relative_precision
end
