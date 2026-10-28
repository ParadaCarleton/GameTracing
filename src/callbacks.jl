using BifurcationKit: BifurcationKit
using Logging: @debug
"""
    CorrectorSettings{E}(; reach, max_kl, max_angle, residual_tolerance, tolerance_floor_ulps, newton_iterations)

The corrector's controls (`CorrectorReach`); `nothing` switches a limit off. `reach`: a corrector further than this
times its step from its predictor is refused. `max_kl`: a corrected point whose largest per-logit probability KL from
the predictor passes this is refused. `max_angle`: so is one whose chord turns past this many radians from the
predictor's direction. `residual_tolerance`, `tolerance_floor_ulps`: the corrector's sup-norm residual, or
`tolerance_floor_ulps` ulps of the largest payoff gap where that is larger (`corrector_tolerance`).
`newton_iterations`: Newton iterations per corrector.
"""
Base.@kwdef struct CorrectorSettings{E <: Real}
    reach::Union{Nothing, E} = 2.0f0
    max_kl::Union{Nothing, E} = nothing
    max_angle::Union{Nothing, E} = nothing
    residual_tolerance::E = 1.0f-5
    tolerance_floor_ulps::E = 32.0f0
    newton_iterations::Int = 10
end

"""
    corrector_tolerance(tolerance, floor_ulps, x)

The residual under which the corrector accepts `x = [w; ρ]`: `tolerance`, or the rounding floor of the
gap residual `w - P G`, `floor_ulps` ulps of the largest payoff gap.
"""
function corrector_tolerance(
    tolerance::T,
    floor_ulps::Real,
    point::AbstractVector,
)::T where {T <: Real}
    largest = maximum(abs, payoff_gaps(point))
    return max(tolerance, convert(T, floor_ulps * eps(typeof(largest)) * largest))
end

"""A trace callback that always lets the trace go on: `(logits, relative_precision)` or `(logits, relative_precision, converged)`."""
struct AcceptAll end

(::AcceptAll)(::AbstractVector, ::Real)::Bool = true
(::AcceptAll)(::AbstractVector, ::Real, ::Bool)::Bool = true

"""Whether `measured` is within `limit`; every measure is within an absent (`nothing`) limit."""
within_limit(::Nothing, ::Real)::Bool = true
within_limit(limit::Real, measured::Real)::Bool = measured <= limit

"""
    CorrectorReach(accept_corrected, settings, geometry, previous, predicted, step_length)

The trace's BifurcationKit `callback_newton`, which also decides the corrector's convergence: a `settled` iterate
ends Newton's method converged (`NewtonConverged`). A non-finite residual stops the corrector,
and so does any iterate further from the step's predictor than `settings.reach` times the predictor's own step, in
`geometry`'s plain norm (`arclength_distance`); the step then halves. On a smooth branch that distance is the branch's
curvature times `ds² / 2`, so halving brings any step under it, while an iterate that lands on another branch is far
from it. A converged closing point is refused the same way past `reach`, past a largest probability KL
`KL(corrected ‖ predicted)` of `settings.max_kl` (`largest_kl`), or past a turn of `settings.max_angle` radians from
the predictor's direction (`chord_angle`, from `previous`). A predictor that did not move has no reach to compare.
`accept_corrected(logits, ρ, converged)` then sees every other closing point. `previous`, `predicted` and
`step_length` hold the current step's start and predictor.
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
end

"""Whether `check` accepts its corrector's converged closing point `corrected` (reach, KL and angle)."""
function closing_accepted(check::CorrectorReach{C, E}, corrected::AbstractVector{E})::Bool where {C, E <: Real}
    (; settings, geometry) = check
    if !within_reach(check, corrected)
        @debug "corrector refused" _group = :trace reason = :reach step_length = check.step_length
        return false
    end
    if !within_limit(settings.max_kl, largest_kl(corrected, check.predicted))
        @debug "corrector refused" _group = :trace reason = :kl
        return false
    end
    if !within_limit(settings.max_angle, chord_angle(check.previous, check.predicted, corrected, geometry))
        @debug "corrector refused" _group = :trace reason = :angle
        return false
    end
    return true
end

"""The residual under which `check`'s corrector accepts `point` (`corrector_tolerance`)."""
function settled_tolerance(check::CorrectorReach, point::AbstractVector)::Real
    return corrector_tolerance(check.settings.residual_tolerance, check.settings.tolerance_floor_ulps, point)
end

function (check::CorrectorReach)(iterate::I;
    fromNewton::Bool = true,
    # fatou-ignore unused-argument: BifurcationKit passes keyword arguments this callback ignores
    keywords...,
)::Union{Bool, BifurcationKit.NewtonConverged} where {I <: NamedTuple}
    if fromNewton
        return newton_verdict(check, iterate, isfinite(iterate.residual))
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
        end
        if !isfinite(iterate.residual)
            @debug "corrector stopped" _group = :trace reason = :nonfinite
            return false
        end
        reached = within_reach(check, iterate.x)
        if !reached
            @debug "corrector stopped" _group = :trace reason = :reach
        end
        return newton_verdict(check, iterate, reached)
    end
    converged = isfinite(iterate.residual) && settled(check, iterate)
    @debug "corrector closed" _group = :trace step = iterate.step residual = iterate.residual converged step_length = check.step_length
    if converged && !closing_accepted(check, iterate.x)
        return false
    end
    if !check.accept_corrected(logits_at(iterate.x), iterate.p, converged)
        return false
    end
    return newton_verdict(check, iterate, true)
end

"""The pin row `ρ - pin` of a corrector `iterate`'s residual: the last entry of Newton's `fx` or of PALC's `res_f`."""
function pin_residual(iterate::NamedTuple)::eltype(iterate.x)
    if haskey(iterate, :fx)
        return last(iterate.fx)
    end
    return last(iterate.res_f)
end

"""
Whether `check`'s corrector has settled `iterate`: its residual is under `corrector_tolerance`, and its relative
precision `ρ` is at the pin to the relative `residual_tolerance`. The pin row needs its own test: the gap floor grows
with the gaps, so at a far target it would pass a `ρ` that never left its start, a point off the map.
"""
function settled(check::CorrectorReach, iterate::I)::Bool where {I <: NamedTuple}
    relative_precision = last(iterate.x)
    offset = pin_residual(iterate)
    pin_scale = max(abs(relative_precision), abs(relative_precision - offset))
    return iterate.residual < settled_tolerance(check, iterate.x) &&
        abs(offset) <= check.settings.residual_tolerance * pin_scale
end

"""
    newton_verdict(check, iterate, proceed)

`check`'s verdict on a Newton `iterate` that its other tests let `proceed`: converged (`NewtonConverged`)
when `settled`, else `proceed`.
"""
function newton_verdict(check::CorrectorReach, iterate::NamedTuple, proceed::T)::Union{
    T,
    BifurcationKit.NewtonConverged,
} where {T <: Bool}
    if proceed && settled(check, iterate)
        return BifurcationKit.NewtonConverged()
    end
    return proceed
end

"""Whether `point` is within `check.settings.reach` times the predictor's step of the predictor."""
function within_reach(check::CorrectorReach, point::AbstractVector)::Bool
    if !(check.step_length > 0)
        return true
    end
    return within_limit(
        scaled_limit(check.settings.reach, check.step_length),
        arclength_distance(check.predicted, point, check.geometry),
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
