using BifurcationKit: BifurcationKit
using ErrorTypes: Err, Ok
using NonlinearSolveFirstOrder: NewtonRaphson
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
end

"""
    CorrectorSettings{E}(; reach, residual_tolerance, tolerance_floor_ulps, newton_iterations, redraw_reach)

The corrector's controls. `reach`: a corrector further than this times its step from its predictor is refused
(`CorrectorReach`). `residual_tolerance`, `tolerance_floor_ulps`: the corrector's sup-norm residual, or
`tolerance_floor_ulps` ulps of the largest logit where that is larger (`corrector_tolerance`).
`newton_iterations`: Newton iterations per corrector. `redraw_reach`: the largest Euclidean distance in logits that a
redraw's fixed-precision correction may move the accepted point (`level_correction`); a farther root is taken for
another branch, and the trace fails `TraceStuck`. The default 0.4 is `StepControl`'s `distance_limit`, Gambit's bound
on one Newton step: a redraw may move the point as far as one corrector step may, and one that moves it farther
changed the branch, not the sample. The test swaps bracket it: the near one moves under 0.2, the far one about 3.5.
"""
Base.@kwdef struct CorrectorSettings{E <: Real}
    reach::E = 2.0f0
    residual_tolerance::E = 1.0f-5
    tolerance_floor_ulps::E = 32.0f0
    newton_iterations::Int = 10
    redraw_reach::E = 0.4f0
end

"""
    TraceSettings{E}(; first_step, min_step, max_step, aggressiveness, state_weight, first_tangent_divisor,
                     corrector, max_steps, tangent, bordered_solver, step_control, orientation_check)

BifurcationKit PALC controls for `trace_endpoint`, in `(w, ρ)` with `w = logits / ρ` the payoff gaps and `ρ` the
relative precision. Step sizes are arclength in PALC's norm `sqrt(θ ‖dw‖² / k + (1 - θ) dρ²)`, `θ` the `state_weight` and `k` the number of logits.

- `first_step`, `min_step`, `max_step`: first, smallest and largest arclength step. A step whose corrector fails
  halves; the trace ends as `TraceStuck` once the step reaches `min_step`.
- `aggressiveness` (BifurcationKit's `a`): a step grows by `1 + a ((N - n) / N)²` after a corrector of `n` of
  its `N = corrector.newton_iterations` iterations.
- `corrector`: a `CorrectorSettings`.
- `state_weight`: the logits' weight against the relative precision's in the arclength norm. `first_tangent_divisor`:
  the first tangent is the secant to a solve at `first_step / first_tangent_divisor`.
- `max_steps`: continuation steps before the trace fails as `NoConvergence`.
- `tangent`: BifurcationKit's tangent algorithm, passed to PALC as is. `Bordered()` (the default) solves the bordered
  Jacobian system. `Secant()` extrapolates the last two points: it can cross onto another arm near a pitchfork that a
  payoff perturbation unfolds, so avoid it there (the test "a pitchfork unfolded by a payoff perturbation").
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
    state_weight::E = 0.5f0
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

"""PALC's `step_control` for `control` over `width` logits with state weight `state_weight`: `nothing` when it is off, else BifurcationKit's `CorrectorQuality`."""
corrector_quality(::Nothing, ::Real, ::Integer)::Nothing = nothing

function corrector_quality(
    control::StepControl{E},
    state_weight::Real,
    width::Integer,
)::BifurcationKit.CorrectorQuality{E} where {E <: Real}
    return BifurcationKit.CorrectorQuality{E}(
        scaled_limit(control.distance_limit, sqrt(state_weight / width)),
        scaled_limit(control.contraction_limit, one(E)),
        scaled_limit(control.growth_limit, one(E)),
        control.contraction_offset,
    )
end

"""BifurcationKit's PALC for `settings` over a trace of `width` logits."""
function palc_algorithm(
    settings::TraceSettings{E},
    width::Integer,
)::BifurcationKit.AbstractContinuationAlgorithm where {E <: Real}
    return BifurcationKit.PALC(;
        tangent = settings.tangent,
        θ = augmented_theta(settings.state_weight, width),
        bls = bordered_linear_solver(settings.bordered_solver, width + 1, E),
        step_control = corrector_quality(
            settings.step_control,
            settings.state_weight,
            width,
        ),
        orientation_check = settings.orientation_check &&
            settings.bordered_solver == MatrixSolve,
        corrector = BifurcationKit.FreeParameterNewton(),
    )
end

"""
BifurcationKit's continuation parameters for `settings`, from relative precision zero to one, with first step
`step_length` and at most `max_steps` steps. Eigenvalues, fold and bifurcation detection serve bifurcation diagrams;
the trace reads only its endpoint.
"""
function continuation_options(
    settings::TraceSettings{E},
    step_length::E,
    max_steps::Integer,
)::BifurcationKit.ContinuationPar where {E <: Real}
    return BifurcationKit.ContinuationPar(;
        ds = step_length,
        dsmin = settings.min_step,
        dsmax = settings.max_step,
        a = settings.aggressiveness,
        η = settings.first_tangent_divisor,
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
        settings.corrector.reach,
        settings.state_weight,
        settings.corrector.residual_tolerance,
        settings.corrector.tolerance_floor_ulps,
        similar(start),
        zero(E),
    )
    opened_continuation!(
        branch_problem(game, start, last(start); record_from_solution = trace_record_precision),
        palc_algorithm(settings, length(start) - 1),
        continuation_options(settings, step_length, max_steps),
        old_tangent,
        finalise,
        corrector,
    )
    return corrector.step_length
end

"""
    opened_continuation!(problem, algorithm, options, old_tangent, finalise, corrector)

Runs BifurcationKit's continuation of `problem`. From the flat start (`old_tangent` is `nothing`) the first tangent
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
)::Nothing
    BifurcationKit.continuation(
        problem,
        algorithm,
        options;
        normC = BifurcationKit.norminf,
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
)::Nothing where {E <: Real}
    start = BifurcationKit.getu0(problem)
    theta = augmented_theta(corrector.theta, length(start) - 1)
    tangent = restart_tangent(finalise.slot.game, start, old_tangent, theta)
    length_in_norm = sqrt(theta * sum(abs2, tangent) / length(tangent) + (1 - theta) * abs2(last(tangent)))
    opening = start .+ (options.ds / length_in_norm) .* tangent
    BifurcationKit.continuation(
        BifurcationKit.ContIterable(
            problem,
            BifurcationKit.update(algorithm, options, nothing),
            options;
            normC = BifurcationKit.norminf,
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
    trace_endpoint(game, width, settings; keep_going, accept_corrected, redraw, level_solver, context)

Follow `game`'s branch of `width` logits from the flat start (zero logits, the only root at precision zero) to
`game.target_precision`, and return `Ok(Traced)` logits there or `Err(TraceFailure)` for the failure that stopped short (a `TraceResult`).
The relative precision `ρ` is the precision in units of the target.

`keep_going(logits, ρ)` runs at each accepted point before the next step's corrector, and `false` stops the trace.
`accept_corrected(logits, ρ, converged)` runs as each corrector ends, and `false` refuses a converged point, which
halves the step. A trace that stops short fails with `NoConvergence` when it spends `settings.max_steps` and `TraceStuck` otherwise (a step below `min_step`, an unchanged accepted point, or a
refusal); a non-finite start or a broken linear solve fails `NotFinite` or `Singular`. `context` rides on
every failure.

`redraw(logits, ρ)` runs at each accepted point that `keep_going` passes, and returns `nothing` or a new game of the
same type and target precision. A new game replaces the map at that point: the point is corrected onto it at fixed
`ρ` by NonlinearSolve's `level_solver` (`level_correction`), and the trace restarts there along the new branch's
tangent, oriented by the old one, with the step it held. A correction that misses fails `TraceStuck` with `projected = true`.
"""
function trace_endpoint(
    game::LogitGame,
    width::Integer,
    settings::TraceSettings{E};
    keep_going::Any = AcceptAll(),
    accept_corrected::Any = AcceptAll(),
    redraw::Any = NoRedraw(),
    level_solver::Any = NewtonRaphson(),
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
        state_of(traced_start)[:],
        zero(E),
        0,
        zero(E),
        zero(E),
        zero(flat),
        zero(E),
        false,
    )
    finalise = TraceFinalise(progress, keep_going, redraw, RedrawSlot(game, false, settings.first_step), 0)
    return redrawn_trace(
        guarded_trace(traced_start, nothing, settings, finalise, accept_corrected),
        finalise,
        settings,
        accept_corrected,
        level_solver,
        context,
    )
end

"""
    redrawn_trace(outcome, finalise, settings, accept_corrected, level_solver, context)

The `TraceResult` of a trace segment that ended with `outcome`. When `redraw` swapped the game there, the last
accepted point is corrected onto the new game (`level_correction`) and the trace restarts from it, a new segment
counted on from `finalise.progress.steps`; a correction that misses fails `TraceStuck` with `projected = true`.
"""
function redrawn_trace(
    outcome::NamedTuple,
    finalise::TraceFinalise,
    settings::TraceSettings{E},
    accept_corrected::Any,
    level_solver::Any,
    context::NamedTuple,
)::TraceResult{E} where {E <: Real}
    (; progress, slot) = finalise
    target_precision = slot.game.target_precision
    if !isnothing(outcome.exception) || !slot.pending || progress.steps >= settings.max_steps
        return reached_state(progress, outcome, one(E), settings.max_steps, target_precision, context)
    end
    slot.pending = false
    correction = level_correction(slot.game, [progress.gaps; progress.relative_precision], settings, level_solver)
    if !correction.corrected
        return Err(TraceFailure(
            TraceStuck,
            (;
                context..., progress.steps, progress.relative_precision, projected = true,
                logits = progress.relative_precision .* progress.gaps,
                correction.residual, correction.distance, correction.retcode,
            ),
        ))
    end
    copyto!(progress.gaps, state_of(correction.point))
    if progress.relative_precision == one(E)
        return reached_state(progress, outcome, one(E), settings.max_steps, target_precision, context)
    end
    next = TraceFinalise(progress, finalise.keep_going, finalise.redraw, slot, progress.steps)
    return redrawn_trace(
        guarded_trace(correction.point, [progress.gap_tangent; progress.tangent_precision], settings, next, accept_corrected),
        next,
        settings,
        accept_corrected,
        level_solver,
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
    reached_state(progress, outcome, target, step_limit, target_precision, context)

The `TraceResult` (`Ok(Traced)` or `Err(TraceFailure)`) of a trace whose `progress` holds the last accepted point and whose `outcome` is the last
corrector's step length or the BifurcationKit exception that ended it.
"""
function reached_state(
    progress::TraceProgress{E},
    outcome::NamedTuple,
    target::E,
    step_limit::Integer,
    target_precision::Real,
    context::NamedTuple,
)::TraceResult{E} where {E <: Real}
    logits = progress.relative_precision .* progress.gaps
    if isnothing(outcome.exception) && progress.relative_precision == target
        return Ok(Traced(logits))
    end
    cause = if uncorrected_start(outcome.exception)
        TraceStuck
    elseif !isnothing(outcome.exception)
        Singular
    elseif progress.steps >= step_limit && !progress.unmoved
        NoConvergence
    else
        TraceStuck
    end
    return Err(TraceFailure(
        cause,
        (;
            context..., progress.steps,
            precision = target_precision * progress.relative_precision,
            progress.precision_min, progress.precision_max, progress.unmoved,
            logits, progress.relative_precision, step_length = E(outcome.step_length),
            logit_tangent = progress.relative_precision .* progress.gap_tangent .+ progress.tangent_precision .* progress.gaps,
            progress.tangent_precision,
            outcome.exception,
        ),
    ))
end
