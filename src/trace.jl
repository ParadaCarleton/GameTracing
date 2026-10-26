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
end

"""
    TraceSettings{E}(; first_step, min_step, max_step, aggressiveness, geometry, first_tangent_divisor,
                     corrector, max_steps, tangent, bordered_solver, step_control, orientation_check)

BifurcationKit PALC controls for `trace_endpoint`, in `(w, ρ)` with `w = logits / ρ` the payoff gaps and `ρ` the
relative precision. Step sizes are arclength in `geometry`'s norm (`StepDistance`); the default is PALC's
`sqrt(θ ‖dw‖² / k + (1 - θ) dρ²)`, `θ` the state weight and `k` the number of gaps.

- `first_step`, `min_step`, `max_step`: first, smallest and largest arclength step. A step whose corrector fails
  halves; the trace ends as `TraceStuck` once the step reaches `min_step`.
- `aggressiveness` (BifurcationKit's `a`): a step grows by `1 + a ((N - n) / N)²` after a corrector of `n` of
  its `N = corrector.newton_iterations` iterations.
- `corrector`: a `CorrectorSettings`.
- `geometry`: a `TraceGeometry`, the norm a trace steps in and its state weight. `first_tangent_divisor`:
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
        scaled_limit(control.distance_limit, sqrt(coordinate_weight(geometry, width))),
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
    geometry = palc_geometry(settings.geometry, width)
    return BifurcationKit.PALC(;
        tangent = settings.tangent,
        θ = geometry.theta,
        dotθ = geometry.dot,
        bls = bordered_linear_solver(settings.bordered_solver, width + 1, E),
        step_control = corrector_quality(settings.step_control, settings.geometry, width),
        orientation_check = settings.orientation_check &&
            settings.bordered_solver == MatrixSolve,
        corrector = BifurcationKit.FreeParameterNewton(),
    )
end

"""
BifurcationKit's continuation parameters for `settings`, from relative precision zero to one. Eigenvalues,
fold and bifurcation detection serve bifurcation diagrams; the trace reads only its endpoint.
"""
function continuation_options(
    settings::TraceSettings{E},
)::BifurcationKit.ContinuationPar where {E <: Real}
    return BifurcationKit.ContinuationPar(;
        ds = settings.first_step,
        dsmin = settings.min_step,
        dsmax = settings.max_step,
        a = settings.aggressiveness,
        η = settings.first_tangent_divisor,
        p_min = zero(E),
        p_max = one(E),
        max_steps = settings.max_steps,
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
    continue_trace(game, traced_start, settings, finalise, accept_corrected)

Runs the PALC continuation of `game`'s branch from `traced_start = [w; 0]` toward relative precision one, with `finalise` at
each accepted point and `accept_corrected` at each corrector, and returns the step length its last corrector tried.
BifurcationKit's constructors are not inferable, so each is built in its own function and passed on without a
named local; it runs once per trace, whose steps dominate it.
"""
function continue_trace(
    game::LogitGame,
    traced_start::AbstractVector{E},
    settings::TraceSettings{E},
    finalise::TraceFinalise,
    accept_corrected::Any,
)::E where {E <: Real}
    corrector = CorrectorReach(
        accept_corrected,
        settings.corrector,
        settings.geometry,
        similar(traced_start),
        similar(traced_start),
        zero(E),
    )
    BifurcationKit.continuation(
        branch_problem(
            game,
            traced_start,
            zero(E);
            record_from_solution = trace_record_precision,
        ),
        palc_algorithm(settings, length(traced_start) - 1),
        continuation_options(settings);
        normC = BifurcationKit.norminf,
        verbosity = 0,
        finalise_solution = finalise,
        callback_newton = corrector,
    )
    return corrector.step_length
end

"""
    trace_endpoint(game, width, settings; keep_going, accept_corrected, context)

Follow `game`'s branch of `width` logits from the flat start (zero logits, the only root at precision zero) to
`game.target_precision`, and return `Ok(Traced)` logits there or `Err(TraceFailure)` for the failure that stopped short (a `TraceResult`).
The relative precision `ρ` is the precision in units of the target.

`keep_going(logits, ρ)` runs at each accepted point before the next step's corrector, and `false` stops the trace.
`accept_corrected(logits, ρ, converged)` runs as each corrector ends, and `false` refuses a converged point, which
halves the step. A trace that stops short fails with `NoConvergence` when it spends `settings.max_steps` and `TraceStuck` otherwise (a step below `min_step`, an unchanged accepted point, or a
refusal); a non-finite start or a broken linear solve fails `NotFinite` or `Singular`. `context` rides on
every failure.
"""
function trace_endpoint(
    game::LogitGame,
    width::Integer,
    settings::TraceSettings{E};
    keep_going::Any = AcceptAll(),
    accept_corrected::Any = AcceptAll(),
    context::NamedTuple = (;),
)::TraceResult{E} where {E <: Real}
    flat = zeros(E, width)
    traced_start = trace_start(game, width, E)
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
    return reached_state(
        progress,
        guarded_trace(
            game,
            traced_start,
            settings,
            TraceFinalise(progress, keep_going),
            accept_corrected,
        ),
        one(E),
        settings.max_steps,
        game.target_precision,
        context,
    )
end

"""
    guarded_trace(game, traced_start, settings, finalise, accept_corrected)

`(; step_length, exception)` of `continue_trace`: the exception is `nothing` for a trace that ran, else the
third-party failure it ended on. BifurcationKit throws a `String` when Newton misses its first point and an
`ErrorException` at its second; a non-finite Jacobian reaches LAPACK. Any other exception propagates.
"""
function guarded_trace(
    game::LogitGame,
    traced_start::AbstractVector{E},
    settings::TraceSettings{E},
    finalise::TraceFinalise,
    accept_corrected::Any,
)::NamedTuple where {E <: Real}
    try
        return (;
            step_length = continue_trace(
                game,
                traced_start,
                settings,
                finalise,
                accept_corrected,
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
