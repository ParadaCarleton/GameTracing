using BifurcationKit: BifurcationKit
using ErrorTypes: Err, Ok
"""
    StepControl{E}(; distance_limit, contraction_limit, growth_limit, contraction_offset)

BifurcationKit's `CorrectorQuality` thresholds, which shrink a step whose corrector looks poor. `distance_limit` is the
Euclidean length of a Newton step's logit part (Gambit's 0.4 in log probabilities); `contraction_limit`, `growth_limit` and
`contraction_offset` bound the residual's contraction and growth across iterations. `nothing` switches a threshold off.
"""
Base.@kwdef struct StepControl{E <: Real}
    distance_limit::Union{Nothing, E} = 0.4f0
    contraction_limit::Union{Nothing, E} = 0.6f0
    growth_limit::Union{Nothing, E} = nothing
    contraction_offset::E = 0.1f0
    # EXPERIMENT: measure Newton steps in Gambit's coordinates (`GambitDistance`), unscaled.
    gambit::Bool = false
end

"""
    TraceSettings{E}(; first_step, min_step, max_step, aggressiveness, geometry, first_tangent_divisor,
                     corrector, max_steps, tangent, bordered_solver, step_control, orientation_check)

BifurcationKit PALC controls for `trace_endpoint`, in `(w, ρ)` with `w = logits / ρ` the payoff gaps and `ρ` the
relative precision. Step sizes are arclength in `geometry`'s norm (`StepDistance`); the default is the Fisher
distance (`TraceGeometry`).

- `first_step`, `min_step`, `max_step`: first, smallest and largest arclength step. A step whose corrector fails
  halves; the trace ends as `TraceStuck` once the step reaches `min_step`.
- `aggressiveness` (BifurcationKit's `a`): a step grows by `1 + a ((N - n) / N)²` after a corrector of `n` of
  its `N = corrector.newton_iterations` iterations.
- `corrector`: a `CorrectorSettings`.
- `geometry`: a `TraceGeometry`, the norm a trace steps in and its state weight. `first_tangent_divisor`:
  the first tangent is the secant to a solve at `first_step / first_tangent_divisor`.
- `max_steps`: continuation steps before the trace fails as `NoConvergence`.
- `tangent`: BifurcationKit's tangent algorithm, passed to PALC as is. `Bordered()` (the default) solves the bordered
  Jacobian system. `Secant()` extrapolates the last two points and saves that solve. Do not use it where a small payoff
  perturbation unfolds a pitchfork: there it crosses onto another arm (the test "a pitchfork unfolded by a payoff
  perturbation"). Over 40 random unfoldings at payoff scale 1e-4 it kept the branch in none, `Bordered()` in all 40.
- `bordered_solver`: a `BorderedSolver`.
- `step_control`: a `StepControl`, or `nothing` for none.
- `orientation_check` orients a bordered tangent by the determinant of the bordered matrix. It takes effect only with
  `MatrixSolve`, the one solver that has a determinant.
"""
Base.@kwdef struct TraceSettings{E <: Real, T <: BifurcationKit.AbstractTangentComputation}
    first_step::E = 0.05f0
    min_step::E = 1.1754944f-38
    max_step::E = 1000.0f0
    aggressiveness::E = 0.5f0
    geometry::TraceGeometry{E} = TraceGeometry{E}()
    first_tangent_divisor::E = 150.0f0
    corrector::CorrectorSettings{E} = CorrectorSettings{E}()
    max_steps::Int = 2_000
    tangent::T = BifurcationKit.Bordered()
    bordered_solver::BorderedSolver = MatrixSolve
    step_control::Union{Nothing, StepControl{E}} = StepControl{E}()
    orientation_check::Bool = true
end

function TraceSettings{E}(;
    tangent::T = BifurcationKit.Bordered(),
    keywords...,
)::TraceSettings{E, T} where {E <: Real, T <: BifurcationKit.AbstractTangentComputation}
    return TraceSettings{E, T}(; tangent, keywords...)
end

"""`limit * scale`, or `nothing` where the threshold is off (`limit` is `nothing`)."""
scaled_limit(::Nothing, ::Real)::Nothing = nothing
scaled_limit(limit::Real, scale::Real)::typeof(limit * scale) = limit * scale

"""
PALC's `step_control` for `control` over `width` gaps in `geometry`: `nothing` when it is off, else BifurcationKit's
`CorrectorQuality`, its distance limit scaled from the gaps' Euclidean length into the norm.
"""
corrector_quality(::Nothing, ::TraceGeometry, ::Integer)::Nothing = nothing

function corrector_quality(
    control::StepControl{E},
    geometry::TraceGeometry{E},
    width::Integer,
)::BifurcationKit.CorrectorQuality{E} where {E <: Real}
    return BifurcationKit.CorrectorQuality{E}(
        scaled_limit(control.distance_limit, control.gambit ? one(E) : sqrt(coordinate_weight(geometry, width))),
        scaled_limit(control.contraction_limit, one(E)),
        scaled_limit(control.growth_limit, one(E)),
        control.contraction_offset,
    )
end

"""
BifurcationKit's PALC for `settings` over a trace of `width` logits to precision `target_precision`, its tangent
oriented by the bordered determinant where `orientation_check` holds (`settings.orientation_check` by default).
"""
function palc_algorithm(
    settings::TraceSettings{E},
    width::Integer,
    target_precision::Real;
    orientation_check::Bool = settings.orientation_check,
    step_distance::Any = nothing,
)::BifurcationKit.AbstractContinuationAlgorithm where {E <: Real}
    return palc_with_geometry(settings, width, palc_geometry(settings.geometry, width, target_precision), orientation_check, step_distance)
end

"""EXPERIMENT: PALC's `step_distance` for `game` under `settings`: `GambitDistance` when its step control is Gambit's."""
function gambit_step_distance(settings::TraceSettings, game::LogitGame)::Any
    if isnothing(settings.step_control) || !settings.step_control.gambit
        return nothing
    end
    return GambitDistance(game.pass!, game.target_precision)
end

"""
BifurcationKit's PALC for `settings` over `width` logits in `geometry` (`palc_geometry`), whose type varies by
distance, with the bordered determinant's `orientation_check` (with `MatrixSolve` only).
"""
function palc_with_geometry(
    settings::TraceSettings{E},
    width::Integer,
    geometry::NamedTuple,
    orientation_check::Bool,
    step_distance::Any = nothing,
)::BifurcationKit.AbstractContinuationAlgorithm where {E <: Real}
    return BifurcationKit.PALC(;
        step_distance,
        tangent = settings.tangent,
        θ = geometry.palc_state_weight,
        dotθ = geometry.dot,
        bls = bordered_linear_solver(settings.bordered_solver, width + 1, E),
        step_control = corrector_quality(settings.step_control, settings.geometry, width),
        orientation_check = orientation_check && settings.bordered_solver == MatrixSolve,
        corrector = BifurcationKit.FreeParameterNewton(),
        orientation_dot = PlainOrientation(settings.geometry),
    )
end

"""
PALC's `orientation_dot`: the plain product `trace_dot` of two branch vectors `[w; ρ]` in `geometry`. Without the
bordered determinant a new tangent takes the sign that keeps this product with the old one positive; the trace's
Fisher weights, frozen at a saturated logit, would let `ρ` alone decide that sign past a fold and turn the trace back.
"""
struct PlainOrientation{E <: Real}
    geometry::TraceGeometry{E}
end

function (orientation::PlainOrientation{E})(
    left::AbstractVector{E},
    right::AbstractVector{E},
    # PALC passes the continuation parameter twice and θ; `trace_dot` reads `ρ` from the vectors.
    ::Real,
    ::Real,
    ::Real,
)::E where {E <: Real}
    return trace_dot(left, right, orientation.geometry)
end

"""
BifurcationKit's continuation parameters for `settings`, from relative precision zero to one, with first step
`step_length`, at most `max_steps` steps and first tangent the secant to a solve at `step_length / first_tangent_divisor`
in the continuation parameter. Eigenvalues, fold and bifurcation detection serve bifurcation diagrams; the trace reads
only its endpoint.
"""
function continuation_options(
    settings::TraceSettings{E},
    step_length::E,
    max_steps::Integer,
    first_tangent_divisor::E,
)::BifurcationKit.ContinuationPar where {E <: Real}
    return BifurcationKit.ContinuationPar(;
        ds = step_length,
        dsmin = settings.min_step,
        dsmax = settings.max_step,
        a = settings.aggressiveness,
        η = first_tangent_divisor,
        p_min = zero(E),
        p_max = one(E),
        max_steps,
        newton_options = BifurcationKit.NewtonPar(;
            tol = settings.corrector.residual_tolerance,
            max_iterations = settings.corrector.newton_iterations,
            α = one(E),
            αmin = one(E) / 1000,
        ),
        save_sol_every_step = 0,
        detect_fold = false,
        detect_bifurcation = 0,
        tol_stability = zero(E),
        dsmin_bisection = zero(E),
        tol_bisection_eigenvalue = zero(E),
        tol_param_bisection_event = zero(E),
    )
end

"""
BifurcationKit's `η` for a trace from the flat start to `target_precision`: its secant solve, `step / η` in the relative
precision, lies `step / settings.first_tangent_divisor` from the start in the trace's distance (`precision_length`).
"""
function opening_divisor(settings::TraceSettings{E}, target_precision::Real)::E where {E <: Real}
    return settings.first_tangent_divisor * precision_length(settings.geometry, zero(E), E(target_precision))
end

"""
    PinScaledNorm(scale)

BifurcationKit's residual norm for a trace: the sup norm, with the pin row `ρ - pin` scaled by `scale`, a unit
relative-precision step's length at the start in the trace's distance (`precision_length`). Unscaled, a far target's
pin row sits under Newton's tolerance whatever `ρ`, so Newton stops before it moves `ρ`.
"""
struct PinScaledNorm{E <: Real}
    scale::E
end

function (norm::PinScaledNorm{E})(residual::AbstractVector{E})::E where {E <: Real}
    return max(maximum(abs, payoff_gaps(residual)), norm.scale * abs(last(residual)))
end

"""
    continue_trace(game, start, old_tangent, settings, finalise, accept_corrected, step_length, max_steps)

Runs the PALC continuation of `game`'s branch from `start = [w; ρ]` toward relative precision one, with `finalise` at
each accepted point and `accept_corrected` at each corrector, and returns the step length its last corrector tried.
From the flat start (`old_tangent` is `nothing`) the first tangent is BifurcationKit's secant; a restart after a
redraw opens along `restart_tangent`, oriented by the trace's `old_tangent`. BifurcationKit's constructors are not
inferable, so each is built in its own function and passed on without a named local; it runs once per segment.
"""
function continue_trace(
    game::LogitGame,
    start::AbstractVector{E},
    old_tangent::Union{Nothing, AbstractVector{E}},
    settings::TraceSettings{E},
    finalise::TraceFinalise,
    accept_corrected::Any,
    step_length::E,
    max_steps::Integer,
)::E where {E <: Real}
    corrector = CorrectorReach(
        accept_corrected,
        settings.corrector,
        settings.geometry,
        similar(start),
        similar(start),
        zero(E),
        GambitDistance(game.pass!, E(game.target_precision)),
    )
    opened_continuation!(
        branch_problem(game, start, last(start); record_from_solution = trace_record_precision),
        palc_algorithm(settings, length(start) - 1, game.target_precision; step_distance = gambit_step_distance(settings, game)),
        continuation_options(settings, step_length, max_steps, opening_divisor(settings, game.target_precision)),
        old_tangent,
        finalise,
        corrector,
        PinScaledNorm(precision_length(settings.geometry, zero(E), E(game.target_precision))),
    )
    return corrector.step_length
end

"""
    opened_continuation!(problem, algorithm, options, old_tangent, finalise, corrector, pin_norm)

Runs BifurcationKit's continuation of `problem`, its residuals measured in `pin_norm` (`PinScaledNorm`). From the flat start (`old_tangent` is `nothing`) the first tangent
is BifurcationKit's secant to a second solve. After a redraw, the run opens through BifurcationKit's two-point start
toward a point `options.ds` along `restart_tangent`, whose secant is that tangent.
"""
function opened_continuation!(
    problem::BifurcationKit.AbstractBifurcationProblem,
    algorithm::BifurcationKit.AbstractContinuationAlgorithm,
    options::BifurcationKit.ContinuationPar,
    ::Nothing,
    finalise::TraceFinalise,
    corrector::CorrectorReach,
    pin_norm::PinScaledNorm,
)::Nothing
    BifurcationKit.continuation(
        problem,
        algorithm,
        options;
        normC = pin_norm,
        verbosity = 0,
        finalise_solution = finalise,
        callback_newton = corrector,
    )
    return nothing
end

function opened_continuation!(
    problem::BifurcationKit.AbstractBifurcationProblem,
    algorithm::BifurcationKit.AbstractContinuationAlgorithm,
    options::BifurcationKit.ContinuationPar,
    old_tangent::AbstractVector{E},
    finalise::TraceFinalise,
    corrector::CorrectorReach,
    pin_norm::PinScaledNorm,
)::Nothing where {E <: Real}
    start = BifurcationKit.getu0(problem)
    dot, palc_state_weight = BifurcationKit.getdot(algorithm), E(BifurcationKit.getθ(algorithm))
    # A Fisher trace's weights are those of the point it restarts from.
    BifurcationKit.reweight!(dot, BifurcationKit.BorderedArray(start, last(start)))
    tangent = restart_tangent(finalise.slot.game, start, old_tangent, dot, palc_state_weight, algorithm.orientation_dot)
    length_in_norm = sqrt(palc_product(dot, palc_state_weight, tangent, tangent))
    # The corrected point's tangent is the new map's; a restart that fails before a step reports it.
    copyto!(finalise.progress.gap_tangent, 1, tangent, 1, length(finalise.progress.gap_tangent))
    finalise.progress.tangent_precision = last(tangent) / length_in_norm
    finalise.progress.gap_tangent ./= length_in_norm
    opening = start .+ (options.ds / length_in_norm) .* tangent
    BifurcationKit.continuation(
        BifurcationKit.ContIterable(
            problem,
            BifurcationKit.update(algorithm, options, nothing),
            options;
            normC = pin_norm,
            verbosity = 0,
            finalise_solution = finalise,
            callback_newton = corrector,
        ),
        start,
        last(start),
        opening,
        last(opening),
    )
    return nothing
end

"""
    trace_endpoint(game, width, settings; keep_going, accept_corrected, redraw, context)

Follow `game`'s branch of `width` logits from the flat start (zero logits, the only root at precision zero) to
`game.target_precision`, and return `Ok(Traced)` logits there or `Err(TraceFailure)` for the failure that stopped short (a `TraceResult`).
The relative precision `ρ` is the precision in units of the target.

`keep_going(logits, ρ)` runs at each accepted point before the next step's corrector, and `false` stops the trace.
`accept_corrected(logits, ρ, converged)` runs as each corrector ends, and `false` refuses a converged point, which
halves the step. A trace that stops short fails with `NoConvergence` when it spends `settings.max_steps` and `TraceStuck` otherwise (a step below `min_step`, an unchanged accepted point, or a
refusal); a non-finite start or a broken linear solve fails `NotFinite` or `Singular`. `context` rides on
every failure.

`redraw(logits, ρ)` runs at each accepted point that `keep_going` passes, and returns `nothing` or a new game of the
same type and target precision. A new game replaces the map at that point: at fixed `ρ` the point is continued
along the homotopy from the old game to the new (`redraw_crossing`), and the trace restarts at its end along the new
branch's tangent, oriented by the old one, with the step it held. A crossing that folds, meets a singular point or
stops short fails `TraceStuck` with `projected = true` and the crossing's `reason`.
"""
function trace_endpoint(
    game::LogitGame,
    width::Integer,
    settings::TraceSettings{E};
    keep_going::Any = AcceptAll(),
    accept_corrected::Any = AcceptAll(),
    redraw::Any = NoRedraw(),
    context::NamedTuple = (;),
)::TraceResult{E} where {E <: Real}
    flat = zeros(E, width)
    traced_start = trace_start(game, flat)
    # The start is a root for every finite game; a non-finite gain shows in its gaps.
    if !all(isfinite, traced_start)
        return Err(TraceFailure(
            NotFinite,
            (; context..., logits = flat, relative_precision = zero(E)),
        ))
    end
    progress = TraceProgress(
        payoff_gaps(traced_start)[:],
        zero(E),
        0,
        zero(E),
        zero(E),
        zero(flat),
        zero(E),
        false,
    )
    finalise = TraceFinalise(progress, keep_going, redraw, RedrawSlot(game, game, false, settings.first_step, gambit_first_step(settings)), 0)
    return redrawn_trace(
        guarded_trace(traced_start, nothing, settings, finalise, accept_corrected),
        finalise,
        settings,
        accept_corrected,
        context,
    )
end

"""
    redrawn_trace(outcome, finalise, settings, accept_corrected, context)

The `TraceResult` of a trace segment that ended with `outcome`. When `redraw` swapped the game there, the last
accepted point crosses onto the new game (`redraw_crossing`) and the trace restarts from it, a new segment counted
on from `finalise.progress.steps`; a crossing that fails ends the trace `TraceStuck` with `projected = true`.
"""
function redrawn_trace(
    outcome::NamedTuple,
    finalise::TraceFinalise,
    settings::TraceSettings{E},
    accept_corrected::Any,
    context::NamedTuple,
)::TraceResult{E} where {E <: Real}
    (; progress, slot) = finalise
    target_precision = slot.game.target_precision
    # A pending swap is crossed first: the crossing takes no trace step, and skipping it would report a root of the
    # old game.
    if !isnothing(outcome.exception) || !slot.pending
        return reached_state(progress, outcome, slot.game, settings, context)
    end
    slot.pending = false
    crossing = redraw_crossing(slot.before, slot.game, [progress.gaps; progress.relative_precision], settings)
    @debug "redraw crossed" _group = :trace reason = crossing.reason steps = crossing.steps
    if crossing.reason != :crossed
        @debug "trace stopped" _group = :trace reason = crossing.reason steps = crossing.steps
        # The context's logits are the accepted point, a root of the game before the swap.
        return Err(TraceFailure(
            TraceStuck,
            (; stopped_context(progress, outcome, target_precision, context)..., projected = true, crossing.reason),
        ))
    end
    copyto!(progress.gaps, payoff_gaps(crossing.point))
    if progress.relative_precision == one(E) || progress.steps >= settings.max_steps
        return reached_state(progress, outcome, slot.game, settings, context)
    end
    next = TraceFinalise(progress, finalise.keep_going, finalise.redraw, slot, progress.steps)
    return redrawn_trace(
        guarded_trace(crossing.point, [progress.gap_tangent; progress.tangent_precision], settings, next, accept_corrected),
        next,
        settings,
        accept_corrected,
        context,
    )
end

"""
    guarded_trace(start, old_tangent, settings, finalise, accept_corrected)

`(; step_length, exception)` of `continue_trace` on `finalise.slot`'s game for the steps `settings.max_steps` leaves:
the exception is `nothing` for a trace that ran, else the third-party failure it ended on. BifurcationKit throws a
`String` when Newton misses its first point and an `ErrorException` at its second; a non-finite Jacobian reaches
LAPACK, and so does a singular restart tangent. Any other exception propagates.
"""
function guarded_trace(
    start::AbstractVector{E},
    old_tangent::Union{Nothing, AbstractVector{E}},
    settings::TraceSettings{E},
    finalise::TraceFinalise,
    accept_corrected::Any,
)::NamedTuple where {E <: Real}
    try
        return (;
            step_length = continue_trace(
                finalise.slot.game,
                start,
                old_tangent,
                settings,
                finalise,
                accept_corrected,
                finalise.slot.step_length,
                settings.max_steps - finalise.step_offset,
            ),
            exception = nothing,
        )
    catch exception
        if !uncorrected_start(exception) && !solver_blowup(exception)
            rethrow()
        end
        return (; step_length = zero(E), exception)
    end
end

"""
    reached_state(progress, outcome, game, settings, context)

The `TraceResult` (`Ok(Traced)` or `Err(TraceFailure)`) of a trace whose `progress` holds the last accepted point and
whose `outcome` is the last corrector's step length or the BifurcationKit exception that ended it. An endpoint at
the target is checked on `game`, the game in effect there: one whose residual misses the corrector tolerance fails
`OffMap`, so no trace returns a root of a game it no longer runs.
"""
function reached_state(
    progress::TraceProgress{E},
    outcome::NamedTuple,
    game::LogitGame,
    settings::TraceSettings{E},
    context::NamedTuple,
)::TraceResult{E} where {E <: Real}
    target_precision = game.target_precision
    if isnothing(outcome.exception) && progress.relative_precision == one(E)
        point = [progress.gaps; one(E)]
        residual = maximum(abs, branch_system!(BranchWorkspace(game, point), game, point, one(E)).residual)
        tolerance = corrector_tolerance(settings.corrector.residual_tolerance, settings.corrector.tolerance_floor_ulps, point)
        if residual < tolerance
            return Ok(Traced(copy(progress.gaps), progress.steps))
        end
        @debug "trace stopped" _group = :trace reason = :off_map residual tolerance
        return Err(TraceFailure(
            OffMap,
            (; stopped_context(progress, outcome, target_precision, context)..., residual, tolerance),
        ))
    end
    # Why the trace stopped short: a start that would not correct, a broken linear solve, a point that did not move,
    # spent steps, or else a step below `min_step`, `keep_going` or a refusal (`STOP_CAUSES`).
    reason = if uncorrected_start(outcome.exception)
        :uncorrected_start
    elseif !isnothing(outcome.exception)
        :solver_blowup
    elseif progress.unmoved
        :unmoved
    elseif progress.steps >= settings.max_steps
        :max_steps
    else
        :stopped_short
    end
    @debug "trace stopped" _group = :trace reason
    return Err(TraceFailure(STOP_CAUSES[reason], stopped_context(progress, outcome, target_precision, context)))
end

# The cause a trace that stops short for each reason fails with.
const STOP_CAUSES = (;
    uncorrected_start = TraceStuck, solver_blowup = Singular, max_steps = NoConvergence, unmoved = TraceStuck,
    stopped_short = TraceStuck,
)

"""
    stopped_context(progress, outcome, target_precision, context)

The context of a trace that stopped at `progress`'s last accepted point after `outcome`: `context`, the steps, the
precision and its range, the logits, the step held, the tangent in logits and the exception.
"""
function stopped_context(
    progress::TraceProgress{E},
    outcome::NamedTuple,
    target_precision::Real,
    context::NamedTuple,
)::NamedTuple where {E <: Real}
    return (;
        context..., progress.steps,
        precision = target_precision * progress.relative_precision,
        progress.precision_min, progress.precision_max, progress.unmoved,
        logits = progress.relative_precision .* progress.gaps, progress.relative_precision,
        step_length = E(outcome.step_length),
        logit_tangent = progress.relative_precision .* progress.gap_tangent .+ progress.tangent_precision .* progress.gaps,
        progress.tangent_precision,
        outcome.exception,
    )
end
