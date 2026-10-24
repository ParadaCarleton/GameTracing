"""
    TraceSettings(; ds, dsmin, dsmax, aggressiveness, corrector_reach, theta, eta, tolerance, floor_ulps,
                  newton_iterations, steps, tangent, bordered_solver, step_control, max_distance,
                  max_contraction, max_growth, contraction_offset, orientation_check)

BifurcationKit PALC controls for `trace_endpoint`, in `(u, t)` with `u = asinh(logits)` and `t` the level.
Steps are arclength in PALC's norm `sqrt(θ ‖du‖² / k + (1 - θ) dt²)`, `k` the number of logits.

- `ds`, `dsmin`, `dsmax`: first, smallest and largest step. A step whose corrector fails halves; the trace ends
  as `TraceStuck` once the step reaches `dsmin`.
- `aggressiveness` (BifurcationKit's `a`): a step grows by `1 + a ((N - n) / N)²` after a corrector of `n` of
  its `N = newton_iterations` iterations.
- `corrector_reach`: a corrector further than this times its step from its predictor is refused (`CorrectorReach`).
- `theta`: the state's weight against the level's in the arclength norm. `eta`: the first tangent is the secant
  to a solve at `ds / eta`.
- `tolerance`, `floor_ulps`: the corrector's sup-norm residual, or `floor_ulps` ulps of the largest logit where
  that is larger (`corrector_tolerance`).
- `steps`: continuation steps before the trace fails as `NoConvergence`.
- `tangent`: `:bordered` solves the bordered Jacobian system at each point, `:secant` extrapolates the last two.
- `bordered_solver`: `:matrix` factors the whole bordered matrix, `:bordering` the Jacobian alone.
- `step_control`: BifurcationKit's `CorrectorQuality` with thresholds `max_distance` (the Euclidean length of a
  Newton step's logit part, as Gambit's 0.4 in log probabilities), `max_contraction`, `max_growth` and
  `contraction_offset`; `Inf` switches a threshold off. `orientation_check` orients a bordered tangent by the
  determinant of the bordered matrix (the `:matrix` solver).
"""
Base.@kwdef struct TraceSettings{E <: Real}
    ds::E = 0.05f0
    dsmin::E = 1.1754944f-38
    dsmax::E = 1000.0f0
    aggressiveness::E = 0.5f0
    corrector_reach::E = 2.0f0
    theta::E = 0.5f0
    eta::E = 150.0f0
    tolerance::E = 1.0f-5
    floor_ulps::E = 32.0f0
    newton_iterations::Int = 10
    steps::Int = 2_000
    tangent::Symbol = :bordered
    bordered_solver::Symbol = :matrix
    step_control::Bool = true
    max_distance::E = 0.4f0
    max_contraction::E = 0.6f0
    max_growth::E = Inf32
    contraction_offset::E = 0.1f0
    orientation_check::Bool = true
end

"""`limit * scale`, or `nothing` where `limit` is infinite: BifurcationKit's `CorrectorQuality` switches a test off with `nothing`."""
function optional_limit(limit::Real, scale::Real)
    if isinf(limit)
        return nothing
    end
    return limit * scale
end

"""PALC's `step_control` for `settings` over `width` logits: `nothing` when it is off, else BifurcationKit's `CorrectorQuality`."""
function step_control(settings::TraceSettings{E}, width::Integer) where {E <: Real}
    if !settings.step_control
        return nothing
    end
    return BifurcationKit.CorrectorQuality{E}(
        optional_limit(settings.max_distance, sqrt(settings.theta / width)),
        optional_limit(settings.max_contraction, one(E)),
        optional_limit(settings.max_growth, one(E)),
        settings.contraction_offset,
    )
end

"""
    continue_trace(game, traced_start, settings, finalise, corrected)

Runs the PALC continuation of `game`'s branch from `traced_start = [u; 0]` toward level one, with `finalise` at
each accepted point and `corrected` at each corrector, and returns the step length its last corrector tried.
Its algorithm, bordered solver and step control are runtime choices of `settings`, so BifurcationKit's
constructors are not inferable; it runs once per trace, whose steps dominate it.
"""
function continue_trace(
    game::LogitGame,
    traced_start::AbstractVector{E},
    settings::TraceSettings,
    finalise::TraceFinalise,
    corrected::Any,
)::E where {E <: Real}
    start = zero(E)
    target = one(E)
    width = length(traced_start) - 1
    problem = branch_problem(
        game,
        traced_start,
        start;
        record_from_solution = trace_record_level,
    )
    newton = BifurcationKit.NewtonPar(;
        tol = E(settings.tolerance),
        max_iterations = settings.newton_iterations,
        α = one(E),
        αmin = one(E) / 1000,
    )
    # Eigenvalues, fold and bifurcation detection serve bifurcation diagrams; the trace reads only its endpoint.
    options = BifurcationKit.ContinuationPar(;
        ds = E(settings.ds),
        dsmin = E(settings.dsmin),
        dsmax = E(settings.dsmax),
        a = E(settings.aggressiveness),
        η = E(settings.eta),
        p_min = start,
        p_max = target,
        max_steps = settings.steps,
        newton_options = newton,
        save_sol_every_step = 0,
        detect_fold = false,
        detect_bifurcation = 0,
        tol_stability = zero(E),
        dsmin_bisection = zero(E),
        tol_bisection_eigenvalue = zero(E),
        tol_param_bisection_event = zero(E),
    )
    tangent = if settings.tangent === :bordered
        BifurcationKit.Bordered()
    else
        BifurcationKit.Secant()
    end
    theta = E(settings.theta)
    algorithm = BifurcationKit.PALC(;
        tangent,
        θ = augmented_theta(theta, width),
        bls = bordered_linear_solver(settings.bordered_solver, length(traced_start), E),
        step_control = step_control(settings, width),
        orientation_check = settings.orientation_check,
        corrector = BifurcationKit.FreeParameterNewton(),
    )
    corrector = CorrectorReach(
        corrected,
        E(settings.corrector_reach),
        theta,
        newton.tol,
        E(settings.floor_ulps),
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
    trace_endpoint(game, flat, settings; accepted, corrected, context)

Follow `game`'s branch from the flat start `(flat, 0)` — at infinite temperature the map is zero and `flat`, the
zero logits, is its only root — to level one, `game.temperature`, and return a `TraceResult` of the logits there
(`sinh` of the traced state). The level is the precision in units of the end's, `t = temperature / τ`.

`accepted(u, t)` runs at each accepted point before the next step's corrector, and `false` stops the trace.
`corrected(u, t, converged)` runs as each corrector ends, and `false` refuses a converged point, which halves
the step. Both see the state without its level. A trace that stops short fails with `NoConvergence` when it
spends `settings.steps` and `TraceStuck` otherwise (a step below `dsmin`, an unchanged accepted point, or a
refusal); a non-finite start or a broken linear solve fails `NotFinite` or `Singular`. `context` rides on
every failure.
"""
function trace_endpoint(
    game::LogitGame,
    flat::AbstractVector{E},
    settings::TraceSettings;
    accepted::Any = AcceptAll(),
    corrected::Any = AcceptAll(),
    context::NamedTuple = (;),
)::TraceResult where {E <: Real}
    start = zero(E)
    target = one(E)
    traced_start = [flat; start]
    # The start is a root of the map for every finite game; a non-finite payoff shows in its residual.
    if !all(isfinite, GameResidual(game)(traced_start, start))
        return TraceResult(Vector{E}(flat), TraceFailure(NotFinite, (; context..., start)))
    end
    progress = TraceProgress(
        Vector{E}(flat),
        start,
        0,
        start,
        start,
        zero(Vector{E}(flat)),
        zero(E),
        false,
    )
    finalise = TraceFinalise(progress, accepted)
    # Third-party code: BifurcationKit throws a `String` when Newton misses its first point and an
    # `ErrorException` at its second; a non-finite Jacobian reaches LAPACK.
    outcome = try
        (; step_length = continue_trace(
            game,
            traced_start,
            settings,
            finalise,
            corrected,
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
        settings.steps,
        game.temperature,
        context,
    )
end

"""
    reached_state(progress, outcome, target, step_limit, temperature, context)

The `TraceResult` of a trace whose `progress` holds the last accepted point and whose `outcome` is the last
corrector's step length or the BifurcationKit exception that ended it.
"""
function reached_state(
    progress::TraceProgress{E},
    outcome::NamedTuple,
    target::E,
    step_limit::Integer,
    temperature::Real,
    context::NamedTuple,
)::TraceResult{E} where {E <: Real}
    logits = sinh.(progress.state)
    if isnothing(outcome.exception) && progress.level == target
        return TraceResult(logits, nothing)
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
    failure = TraceFailure(
        cause,
        (;
            context..., progress.steps,
            temperature = level_temperature(temperature, progress.level),
            progress.level_min, progress.level_max, progress.unmoved,
            state = copy(progress.state), progress.level, step_length = E(
                outcome.step_length,
            ),
            tangent_state = copy(progress.tangent_state), progress.tangent_level,
            outcome.exception,
        ),
    )
    return TraceResult(logits, failure)
end
