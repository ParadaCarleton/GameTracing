"""The distance from `from` to `to`, both `[u; t]`, in PALC's norm `sqrt(θ ‖Δu‖² / k + (1 - θ) Δt²)`."""
function arclength_distance(
    from::AbstractVector{E},
    to::AbstractVector{E},
    theta::E,
)::E where {E <: Real}
    state_squares = sum(
        abs2(finish - start)
        for (start, finish) in zip(state_of(from), state_of(to))
    )
    return sqrt(
        theta * state_squares / (length(to) - 1) +
            (1 - theta) * abs2(last(to) - last(from)),
    )
end

"""
    corrector_tolerance(tolerance, floor_ulps, x)

The residual under which the corrector accepts `x = [u; t]`: `tolerance`, or the rounding floor of
`z - T(z)` at the logits `z = sinh(u)`, `floor_ulps` ulps of the largest.
"""
function corrector_tolerance(
    tolerance::T,
    floor_ulps::Real,
    x::AbstractVector,
)::T where {T <: Real}
    largest = maximum(entry -> abs(sinh(primal(entry))), state_of(x))
    return max(tolerance, convert(T, floor_ulps * eps(typeof(largest)) * largest))
end

"""A trace callback that always lets the trace go on: `(logits, relative_precision)` or `(logits, relative_precision, converged)`."""
struct AcceptAll end

(::AcceptAll)(::AbstractVector, ::Real)::Bool = true
(::AcceptAll)(::AbstractVector, ::Real, ::Bool)::Bool = true

"""
    CorrectorReach(accept_corrected, reach, theta, tolerance, floor_ulps, predicted, step_length)

The trace's BifurcationKit `callback_newton`, which also decides the corrector's convergence: an iterate
under `corrector_tolerance` ends Newton's method converged (`NewtonConverged`). A non-finite residual stops
the corrector, and so does any iterate further from the step's predictor than `reach` times the predictor's
own step, in PALC's norm (`arclength_distance`); the step then halves. A converged closing point beyond that
distance is refused the same way. On a smooth branch the distance is the branch's curvature times `ds² / 2`,
so halving brings any step under it, while an iterate that diverges or lands on another branch is far from it.
A predictor that did not move has no reach to compare. `accept_corrected(u, t, converged)` then sees every other
closing point. `predicted` and `step_length` hold the current step's predictor.
"""
mutable struct CorrectorReach{C, E <: Real}
    # Mutable because BifurcationKit shows the predictor only to a corrector's first callback and the
    # accept_corrected point only to its last, and returns neither.
    accept_corrected::C
    reach::E
    theta::E
    tolerance::E
    floor_ulps::E
    predicted::Vector{E}
    step_length::E
end

function (check::CorrectorReach)(
    iterate::NamedTuple;
    fromNewton::Bool = true,
    keywords...,
)::Union{Bool, BifurcationKit.NewtonConverged}
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
            check.step_length = arclength_distance(iterate.z0.u, iterate.x, check.theta)
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
    converged = isfinite(iterate.residual) &&
        iterate.residual < corrector_tolerance(check.tolerance, check.floor_ulps, iterate.x)
    @debug "corrector closed" _group = :trace step = iterate.step residual = iterate.residual converged step_length = check.step_length
    if converged && !within_reach(check, iterate.x)
        @debug "corrector reach refused" _group = :trace step_length = check.step_length distance = arclength_distance(
            check.predicted,
            iterate.x,
            check.theta,
        )
        return false
    end
    if !check.accept_corrected(sinh.(state_of(iterate.x)), iterate.p, converged)
        return false
    end
    return newton_verdict(check, iterate, true)
end

"""
    newton_verdict(check, iterate, proceed)

`check`'s verdict on a Newton `iterate` that its other tests let `proceed`: converged (`NewtonConverged`)
under `corrector_tolerance`, else `proceed`.
"""
function newton_verdict(check::CorrectorReach, iterate::NamedTuple, proceed::T)::Union{
    T,
    BifurcationKit.NewtonConverged,
} where {T <: Bool}
    if proceed &&
            iterate.residual <
                corrector_tolerance(check.tolerance, check.floor_ulps, iterate.x)
        return BifurcationKit.NewtonConverged()
    end
    return proceed
end

"""Whether `point` is within `check.reach` times the predictor's step of the predictor."""
function within_reach(check::CorrectorReach, point::AbstractVector)::Bool
    if !(check.step_length > 0)
        return true
    end
    return arclength_distance(check.predicted, point, check.theta) <=
        check.reach * check.step_length
end

"""
    TraceProgress(asinh_logits, relative_precision, steps, precision_min, precision_max, asinh_tangent, tangent_precision, unmoved)

The last accepted point `(asinh_logits, relative_precision)` of a trace, its step count, the range of the relative precision
over accepted points and the tangent there (both in PALC's `asinh` coordinates): BifurcationKit's `finalise_solution`
callback. An accepted point bitwise equal to the one before it (a step below the point's rounding) sets `unmoved` and
stops the trace, a stall. `JULIA_DEBUG=trace` lists every point.
"""
mutable struct TraceProgress{E <: Real}
    asinh_logits::Vector{E}
    relative_precision::E
    steps::Int
    precision_min::E
    precision_max::E
    asinh_tangent::Vector{E}
    tangent_precision::E
    unmoved::Bool
end

function (progress::TraceProgress)(
    point::Any,
    tangent::Any,
    step::Integer,
    result::Any;
    keywords...,
)::Bool
    @debug "trace point" _group = :trace step relative_precision = point.p x = copy(
        point.u,
    ) tangent = copy(tangent.u)
    progress.unmoved = point.p == progress.relative_precision &&
        view(point.u, eachindex(progress.asinh_logits)) == progress.asinh_logits
    copyto!(progress.asinh_logits, 1, point.u, 1, length(progress.asinh_logits))
    progress.relative_precision = point.p
    progress.steps = step
    progress.precision_min = min(progress.precision_min, point.p)
    progress.precision_max = max(progress.precision_max, point.p)
    copyto!(progress.asinh_tangent, 1, tangent.u, 1, length(progress.asinh_tangent))
    progress.tangent_precision = tangent.p
    return !progress.unmoved
end

"""
    TraceFinalise(progress, keep_going)

BifurcationKit's `finalise_solution` of a trace: `progress` records the point, and the trace goes on
while `keep_going(logits, relative_precision)` holds.
"""
struct TraceFinalise{R <: TraceProgress, A}
    progress::R
    keep_going::A
end

function (finalise::TraceFinalise)(
    point::Any,
    tangent::Any,
    step::Integer,
    result::Any;
    keywords...,
)::Bool
    recorded = finalise.progress(point, tangent, step, result)
    return recorded && finalise.keep_going(sinh.(state_of(point.u)), point.p)
end

"""One scalar per step for BifurcationKit's branch record, the relative_precision: `TraceProgress` keeps the point."""
function trace_record_precision(::Any, relative_precision::L; keywords...)::L where {L <:
    Real}
    return relative_precision
end
