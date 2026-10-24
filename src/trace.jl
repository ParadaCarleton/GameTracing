"""How PALC finds the tangent at each point: `BorderedTangent` solves the bordered Jacobian system, `SecantTangent` extrapolates the last two points."""
@enum TangentMethod BorderedTangent SecantTangent

"""
    StepControl{E}(; max_distance, max_contraction, max_growth, contraction_offset)

BifurcationKit's `CorrectorQuality` thresholds, which shrink a step whose corrector looks poor. `max_distance` is the
Euclidean length of a Newton step's logit part (Gambit's 0.4 in log probabilities); `max_contraction`, `max_growth` and
`contraction_offset` bound the residual's contraction and growth across iterations. `nothing` switches a threshold off.
"""
Base.@kwdef struct StepControl{E <: Real}
    max_distance::Union{Nothing, E} = 0.4f0
    max_contraction::Union{Nothing, E} = 0.6f0
    max_growth::Union{Nothing, E} = nothing
    contraction_offset::E = 0.1f0
end

"""
    TraceSettings{E}(; first_step, min_step, max_step, aggressiveness, corrector_reach, state_weight,
                     first_tangent_divisor, residual_tolerance, tolerance_floor_ulps, max_newton_iterations,
                     max_steps, tangent, bordered_solver, step_control, orientation_check)

BifurcationKit PALC controls for `trace_endpoint`, in `(u, ρ)` with `u = asinh(logits)` and `ρ` the relative precision.
Step sizes are arclength in PALC's norm `sqrt(θ ‖du‖² / k + (1 - θ) dρ²)`, `θ` the `state_weight` and `k` the number of logits.

- `first_step`, `min_step`, `max_step`: first, smallest and largest arclength step. A step whose corrector fails
  halves; the trace ends as `TraceStuck` once the step reaches `min_step`.
- `aggressiveness` (BifurcationKit's `a`): a step grows by `1 + a ((N - n) / N)²` after a corrector of `n` of
  its `N = max_newton_iterations` iterations.
- `corrector_reach`: a corrector further than this times its step from its predictor is refused (`CorrectorReach`).
- `state_weight`: the logits' weight against the relative precision's in the arclength norm. `first_tangent_divisor`:
  the first tangent is the secant to a solve at `first_step / first_tangent_divisor`.
- `residual_tolerance`, `tolerance_floor_ulps`: the corrector's sup-norm residual, or `tolerance_floor_ulps` ulps of
  the largest logit where that is larger (`corrector_tolerance`).
- `max_steps`: continuation steps before the trace fails as `NoConvergence`.
- `tangent`: a `TangentMethod`. `bordered_solver`: a `BorderedSolver`.
- `step_control`: a `StepControl`, or `nothing` for none.
- `orientation_check` orients a bordered tangent by the determinant of the bordered matrix. It takes effect only with
  `MatrixSolve`, the one solver that has a determinant.
"""
Base.@kwdef struct TraceSettings{E <: Real}
    first_step::E = 0.05f0
    min_step::E = 1.1754944f-38
    max_step::E = 1000.0f0
    aggressiveness::E = 0.5f0
    corrector_reach::E = 2.0f0
    state_weight::E = 0.5f0
    first_tangent_divisor::E = 150.0f0
    residual_tolerance::E = 1.0f-5
    tolerance_floor_ulps::E = 32.0f0
    max_newton_iterations::Int = 10
    max_steps::Int = 2_000
    tangent::TangentMethod = BorderedTangent
    bordered_solver::BorderedSolver = MatrixSolve
    step_control::Union{Nothing, StepControl{E}} = StepControl{E}()
    orientation_check::Bool = true
end

"""`limit * scale`, or `nothing` where the threshold is off (`limit` is `nothing`)."""
function scaled_limit(limit::Union{Nothing, Real}, scale::Real)
    if isnothing(limit)
        return nothing
    end
    return limit * scale
end

"""PALC's `step_control` for `control` over `width` logits with state weight `state_weight`: `nothing` when it is off, else BifurcationKit's `CorrectorQuality`."""
function corrector_quality(::Nothing, ::Real, ::Integer)
    return nothing
end

function corrector_quality(
    control::StepControl{E},
    state_weight::Real,
    width::Integer,
)::BifurcationKit.CorrectorQuality{E} where {E <: Real}
    return BifurcationKit.CorrectorQuality{E}(
        scaled_limit(control.max_distance, sqrt(state_weight / width)),
        scaled_limit(control.max_contraction, one(E)),
        scaled_limit(control.max_growth, one(E)),
        control.contraction_offset,
    )
end

"""
    continue_trace(game, traced_start, settings, finalise, accept_corrected)

Runs the PALC continuation of `game`'s branch from `traced_start = [u; 0]` toward relative precision one, with `finalise` at
each accepted point and `accept_corrected` at each corrector, and returns the step length its last corrector tried.
Its algorithm, bordered solver and step control are runtime choices of `settings`, so BifurcationKit's
constructors are not inferable; it runs once per trace, whose steps dominate it.
"""
function continue_trace(
    game::LogitGame,
    traced_start::AbstractVector{E},
    settings::TraceSettings,
    finalise::TraceFinalise,
    accept_corrected::Any,
)::E where {E <: Real}
    start = zero(E)
    target = one(E)
    width = length(traced_start) - 1
    problem = branch_problem(
        game,
        traced_start,
        start;
        record_from_solution = trace_record_precision,
    )
    newton = BifurcationKit.NewtonPar(;
        tol = E(settings.residual_tolerance),
        max_iterations = settings.max_newton_iterations,
        α = one(E),
        αmin = one(E) / 1000,
    )
    # Eigenvalues, fold and bifurcation detection serve bifurcation diagrams; the trace reads only its endpoint.
    options = BifurcationKit.ContinuationPar(;
        ds = E(settings.first_step),
        dsmin = E(settings.min_step),
        dsmax = E(settings.max_step),
        a = E(settings.aggressiveness),
        η = E(settings.first_tangent_divisor),
        p_min = start,
        p_max = target,
        max_steps = settings.max_steps,
        newton_options = newton,
        save_sol_every_step = 0,
        detect_fold = false,
        detect_bifurcation = 0,
        tol_stability = zero(E),
        dsmin_bisection = zero(E),
        tol_bisection_eigenvalue = zero(E),
        tol_param_bisection_event = zero(E),
    )
    tangent = if settings.tangent == BorderedTangent
        BifurcationKit.Bordered()
    else
        BifurcationKit.Secant()
    end
    state_weight = E(settings.state_weight)
    algorithm = BifurcationKit.PALC(;
        tangent,
        θ = augmented_theta(state_weight, width),
        bls = bordered_linear_solver(settings.bordered_solver, length(traced_start), E),
        step_control = corrector_quality(settings.step_control, state_weight, width),
        orientation_check = settings.orientation_check &&
            settings.bordered_solver == MatrixSolve,
        corrector = BifurcationKit.FreeParameterNewton(),
    )
    corrector = CorrectorReach(
        accept_corrected,
        E(settings.corrector_reach),
        state_weight,
        newton.tol,
        E(settings.tolerance_floor_ulps),
        similar(traced_start),
        zero(E),
    )
    BifurcationKit.continuation(
        problem,
        algorithm,
        options;
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
`game.target_precision`, and return `Traced` logits there or the `TraceFailure` that stopped short (a `TraceResult`).
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
    start = zero(E)
    target = one(E)
    flat = zeros(E, width)
    traced_start = [flat; start]
    # The start is a root of the map for every finite game; a non-finite payoff shows in its residual.
    if !all(isfinite, GameResidual(game)(traced_start, start))
        return TraceFailure(
            NotFinite,
            (; context..., logits = flat, relative_precision = start),
        )
    end
    progress = TraceProgress(copy(flat), start, 0, start, start, zero(flat), zero(E), false)
    finalise = TraceFinalise(progress, keep_going)
    # Third-party code: BifurcationKit throws a `String` when Newton misses its first point and an
    # `ErrorException` at its second; a non-finite Jacobian reaches LAPACK.
    outcome = try
        (; step_length = continue_trace(
            game,
            traced_start,
            settings,
            finalise,
            accept_corrected,
        )::E, exception = nothing)
    catch exception
        if !uncorrected_start(exception) && !solver_blowup(exception)
            rethrow()
        end
        (; step_length = zero(E), exception)
    end
    return reached_state(
        progress,
        outcome,
        target,
        settings.max_steps,
        game.target_precision,
        context,
    )
end

"""
    reached_state(progress, outcome, target, step_limit, target_precision, context)

The `TraceResult` (`Traced` or `TraceFailure`) of a trace whose `progress` holds the last accepted point and whose `outcome` is the last
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
    logits = sinh.(progress.asinh_logits)
    if isnothing(outcome.exception) && progress.relative_precision == target
        return Traced(logits)
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
    return TraceFailure(
        cause,
        (;
            context..., progress.steps,
            precision = target_precision * progress.relative_precision,
            progress.precision_min, progress.precision_max, progress.unmoved,
            logits, progress.relative_precision, step_length = E(outcome.step_length),
            asinh_tangent = copy(progress.asinh_tangent), progress.tangent_precision,
            outcome.exception,
        ),
    )
end
