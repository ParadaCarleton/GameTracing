using BifurcationKit: BifurcationKit
using ErrorTypes: Err, Ok
using LinearAlgebra: LU
"""
    StepControl{E}(; distance_limit, growth_limit, turn_limit)

BifurcationKit's `CorrectorQuality` thresholds, which shrink a step whose corrector looks poor. `distance_limit` is the
Euclidean length of a Newton step's logit part (Gambit's 0.4 in log probabilities); `growth_limit` bounds a step's
growth (Gambit's `max_accel` 1.1), and the step then shrinks as the distance or contraction nears its limit.
`turn_limit` refuses a step whose tangent turns this many radians from the last one, Allgower and Georg's
test (6.1.10) for a step across a fold onto another branch; 1 is under Gambit's quarter turn (`t·newT < 0`, path.cc).
The corrector's contraction rate (`CorrectorSettings`) also slows growth. `nothing` switches a threshold off.
"""
Base.@kwdef struct StepControl{E <: Real}
    distance_limit::Union{Nothing, E} = 0.4f0
    growth_limit::Union{Nothing, E} = 1.1f0
    turn_limit::Union{Nothing, E} = 1.0f0
end

"""
    TraceSettings{E}(; first_step, max_step, aggressiveness, geometry, first_tangent_divisor,
                     corrector, max_steps, tangent, bordered_solver, step_control, orientation_check, palc_corrector,
                     palc_options, continuation_options)

BifurcationKit PALC controls for `trace_endpoint`, in `(w, ρ)` with `w = logits / ρ` the payoff gaps and `ρ` the
relative precision. Step sizes are arclength in `geometry`'s norm (`StepDistance`); the default is the Fisher
distance (`TraceGeometry`).

- `first_step`, `max_step`: first and largest arclength step. A step whose corrector fails halves; the trace ends as
  `TraceStuck` once its predictor no longer moves the point at the float type's resolution (`step_unresolved`).
  BifurcationKit's `dsmin` is `floatmin(E)`, a guard only: a subnormal step gives NaN. After the predictor stops
  moving, BifurcationKit still halves to that guard, about 100 halvings of one refused residual evaluation each.
- `aggressiveness` (BifurcationKit's `a`): without a `step_control` growth limit, a step grows by
  `1 + a ((N - n) / N)²` after a corrector of `n` of its `N = corrector.newton_iterations` iterations.
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
- `palc_corrector`: PALC's `corrector` keyword, `nothing` by default, which is BifurcationKit's own Newton method
  (`newton_palc`; it stops on leaving `[0, 1]`). `BifurcationKit.FreeParameterNewton()` leaves `ρ` free there instead,
  the trace's choice before this field. `CorrectorReach` reads any corrector's Newton callbacks.
- `palc_options`: a `NamedTuple` of further keywords for `BifurcationKit.PALC`, which override the trace's own
  (`tangent`, `θ`, `dotθ`, `bls`, `step_control`, `orientation_check`, `corrector`, `orientation_dot`). Do not override
  `step_control`, `orientation_check`, `orientation_dot`, `θ`, `dotθ` or `bls`: the trace's step shrinking, tangent orientation
  and geometry rest on them, and `TraceSettings` fields set them consistently.
- `continuation_options`: a `NamedTuple` of further keywords for `BifurcationKit.ContinuationPar`, which override the trace's
  own. Do not override `newton_options`, `p_min`, `p_max`, `ds` or the `dsmin`/`dsmax` pair: `CorrectorReach` converges a
  corrector on its own test and needs `tol = 0`, and the trace ends at `p_max`.
"""
Base.@kwdef struct TraceSettings{
        E <: Real,
        T <: BifurcationKit.AbstractTangentComputation,
        C,
        P <: NamedTuple,
        O <: NamedTuple,
    }
    first_step::E = 0.05f0
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
    palc_corrector::C = nothing
    palc_options::P = (;)
    continuation_options::O = (;)
end

function TraceSettings{E}(;
    tangent::T = BifurcationKit.Bordered(),
    palc_corrector::C = nothing,
    palc_options::P = (;),
    continuation_options::O = (;),
    keywords...,
)::TraceSettings{E, T, C, P, O} where {
        E <: Real, T <: BifurcationKit.AbstractTangentComputation, C, P <: NamedTuple, O <: NamedTuple,
    }
    return TraceSettings{E, T, C, P, O}(;
        tangent, palc_corrector, palc_options, continuation_options, keywords...,
    )
end

"""`limit * scale`, or `nothing` where the threshold is off (`limit` is `nothing`)."""
scaled_limit(::Nothing, ::Real)::Nothing = nothing
scaled_limit(limit::Real, scale::Real)::typeof(limit * scale) = limit * scale

"""
PALC's `step_control` for `control` over `width` gaps in `geometry`: `nothing` when it is off, else BifurcationKit's
`CorrectorQuality`, its distance limit scaled from the gaps' Euclidean length into the norm. Its contraction rate,
`corrector`'s limit and offset, slows growth only: `CorrectorReach` refuses on the rate in logits. A path that is not
oriented has no turn test: a solver without the bordered determinant that BifurcationKit's turn of a `Bordered`
tangent needs.
"""
corrector_quality(
    ::Nothing,
    ::CorrectorSettings,
    ::TraceGeometry,
    ::Integer,
    ::Bool,
)::Nothing = nothing

function corrector_quality(
    control::StepControl{E},
    corrector::CorrectorSettings{E},
    geometry::TraceGeometry{E},
    width::Integer,
    oriented::Bool,
)::BifurcationKit.CorrectorQuality{E} where {E <: Real}
    return BifurcationKit.CorrectorQuality{E}(
        scaled_limit(control.distance_limit, sqrt(coordinate_weight(geometry, width))),
        scaled_limit(corrector.contraction_limit, one(E)),
        scaled_limit(control.growth_limit, one(E)),
        corrector.contraction_offset,
        scaled_limit(ifelse(oriented, control.turn_limit, nothing), one(E)),
        false,
    )
end

"""
PALC's `orientation_dot`: the plain product `trace_dot` of two branch vectors `[w; ρ]` in `geometry`; `ρ` is in
the vectors, so the continuation parameters add nothing. Without the bordered determinant a new tangent takes the
sign that keeps this product with the old one positive; the trace's Fisher weights, frozen at a saturated logit,
would let `ρ` alone decide that sign past a fold and turn the trace back.
"""
struct PlainOrientation{E <: Real}
    geometry::TraceGeometry{E}
end

function (orientation::PlainOrientation{E})(
    left::AbstractVector{E},
    right::AbstractVector{E},
    # PALC passes the continuation parameters and its state weight θ too.
    ::Real,
    ::Real,
    ::Real,
)::E where {E <: Real}
    return trace_dot(left, right, orientation.geometry)
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
)::BifurcationKit.AbstractContinuationAlgorithm where {E <: Real}
    return palc_with_geometry(
        settings,
        width,
        palc_geometry(settings.geometry, width, target_precision),
        orientation_check,
        PlainOrientation(settings.geometry),
    )
end

"""
BifurcationKit's PALC for `settings` over `width` logits in `geometry` (`palc_geometry`), whose type varies by
distance, with the bordered determinant's `orientation_check` (with `MatrixSolve` only) and `orientation`.
"""
function palc_with_geometry(
    settings::TraceSettings{E},
    width::Integer,
    geometry::NamedTuple,
    orientation_check::Bool,
    orientation::PlainOrientation{E},
)::BifurcationKit.AbstractContinuationAlgorithm where {E <: Real}
    oriented = orientation_check && settings.bordered_solver == MatrixSolve
    own = (;
        tangent = settings.tangent,
        θ = geometry.palc_state_weight,
        dotθ = geometry.dot,
        bls = bordered_linear_solver(settings.bordered_solver, width + 1, E),
        step_control = corrector_quality(
            settings.step_control,
            settings.corrector,
            settings.geometry,
            width,
            oriented,
        ),
        orientation_check = oriented,
        corrector = settings.palc_corrector,
        orientation_dot = orientation,
    )
    return BifurcationKit.PALC(; merge(own, settings.palc_options)...)
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
    own = (;
        ds = step_length,
        dsmin = floatmin(E),
        dsmax = settings.max_step,
        a = settings.aggressiveness,
        η = first_tangent_divisor,
        p_min = zero(E),
        p_max = one(E),
        max_steps,
        newton_options = BifurcationKit.NewtonPar(;
            # Only `CorrectorReach` converges a corrector (`settled`), and an exact root.
            tol = zero(E),
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
    return BifurcationKit.ContinuationPar(; merge(own, settings.continuation_options)...)
end

"""
BifurcationKit's `η` for a trace from the flat start to `target_precision`: its secant solve, `step / η` in the relative
precision, lies `step / settings.first_tangent_divisor` from the start in the trace's distance (`precision_length`).
"""
function opening_divisor(settings::TraceSettings{E}, target_precision::Real)::E where {E <:
    Real}
    return settings.first_tangent_divisor *
        precision_length(settings.geometry, zero(E), E(target_precision))
end

"""
    trace_iterator(game, start, settings, finalise, corrector, step_length, max_steps)

BifurcationKit's PALC iterator over `game`'s branch from `start = [w; ρ]` toward relative precision one, with
`finalise` at each accepted point, `corrector` (`CorrectorReach`) at each Newton iterate and a branch of
`trace_record_point` records. BifurcationKit's constructors are not inferable, so each is built in its own function
and passed on without a named local; it runs once per segment.
"""
function trace_iterator(
    game::LogitGame,
    start::AbstractVector{E},
    settings::TraceSettings{E},
    finalise::TraceFinalise,
    corrector::CorrectorReach,
    step_length::E,
    max_steps::Integer,
)::BifurcationKit.ContIterable where {E <: Real}
    return continuation_iterator(
        branch_problem(game, start, last(start); record_from_solution = trace_record_point),
        palc_algorithm(settings, length(start) - 1, game.target_precision),
        continuation_options(
            settings,
            step_length,
            max_steps,
            opening_divisor(settings, game.target_precision),
        ),
        finalise,
        corrector,
    )
end

function continuation_iterator(
    problem::BifurcationKit.AbstractBifurcationProblem,
    algorithm::BifurcationKit.AbstractContinuationAlgorithm,
    options::BifurcationKit.ContinuationPar,
    finalise::TraceFinalise,
    corrector::CorrectorReach,
)::BifurcationKit.ContIterable
    return BifurcationKit.ContIterable(
        problem,
        BifurcationKit.update(algorithm, options, nothing),
        options;
        normC = BifurcationKit.norminf,
        verbosity = 0,
        finalise_solution = finalise,
        callback_newton = corrector,
    )
end

"""
    opened_state(iterator, ::Nothing) / opened_state(iterator, opening)

BifurcationKit's first continuation state of `iterator`. From the flat start the first tangent is BifurcationKit's
secant to a second solve. After a redraw, the run opens through BifurcationKit's two-point start toward the point
`opening`, whose secant is the restart tangent.
"""
opened_state(
    iterator::BifurcationKit.ContIterable,
    ::Nothing,
)::BifurcationKit.ContState = first(iterate(iterator))

function opened_state(
    iterator::BifurcationKit.ContIterable,
    opening::AbstractVector,
)::BifurcationKit.ContState
    start = BifurcationKit.getu0(iterator.prob)
    return first(BifurcationKit.iterate_from_two_points(
        iterator,
        start,
        last(start),
        opening,
        last(opening),
    ))
end

"""
    restart_opening(iterator, factorization, old_tangent)

`(; opening, gap_tangent, tangent_precision)` of a restart of `iterator` after a redraw: the point `opening` `ds`
along `restart_tangent` from the swapped game's `factorization`, oriented by the trace's `old_tangent`, and that
tangent of unit length in the trace's norm, split into its gap and precision parts.
"""
function restart_opening(
    iterator::BifurcationKit.ContIterable,
    factorization::LU{E},
    old_tangent::AbstractVector{E},
)::NamedTuple where {E <: Real}
    algorithm = iterator.alg
    start = BifurcationKit.getu0(iterator.prob)
    dot, palc_state_weight = BifurcationKit.getdot(algorithm),
        E(BifurcationKit.getθ(algorithm))
    # A Fisher trace's weights are those of the point it restarts from.
    BifurcationKit.reweight!(dot, BifurcationKit.BorderedArray(start, last(start)))
    tangent = restart_tangent(factorization, old_tangent, palc_state_weight, algorithm.orientation_dot)
    length_in_norm = sqrt(palc_product(dot, palc_state_weight, tangent, tangent))
    return (;
        opening = start .+ (iterator.contparams.ds / length_in_norm) .* tangent,
        gap_tangent = tangent[begin:(end - 1)] ./ length_in_norm,
        tangent_precision = last(tangent) / length_in_norm,
    )
end

"""
    trace_endpoint(game, width, settings; keep_going, accept_corrected, redraw, context)

Follow `game`'s branch of `width` logits from the flat start (zero logits, the only root at precision zero) to
`game.target_precision`, and return `Ok(Traced)` logits there or `Err(TraceFailure)` for the failure that stopped short (a `TraceResult`).
The relative precision `ρ` is the precision in units of the target.

`keep_going(logits, ρ)` runs at each accepted point before the next step's corrector, and `false` stops the trace.
`accept_corrected(logits, ρ, converged)` runs as each corrector ends, and `false` refuses a converged point, which
halves the step. A trace that stops short fails with `NoConvergence` when it spends `settings.max_steps` and `TraceStuck` otherwise (a step below the float type's resolution, an unchanged accepted point, or a
refusal); a non-finite start or a broken linear solve fails `NotFinite` or `Singular`. `context` rides on
every failure.

`redraw(logits, ρ, shift)` runs at each accepted point short of `ρ = 1` that `keep_going` passes, and returns
`nothing`, a new game of the same type and target precision, or the `ShiftedGame` that `shift(game)` (`RootShift`)
returned for one. That check tells whether a candidate game's root lies within the corrector's reach, so a caller can
refine a draw that does not before it returns; the restart reuses its factorization of the new game's Jacobian. A new game replaces the map for the steps after that point: the
trace restarts there along the new map's tangent, oriented by the old one, with the step it held, and its next
corrector lands on the new map under the same step control. The endpoint, with no step after it, is never redrawn.
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
    opening = TraceProgress(
        payoff_gaps(traced_start)[:],
        zero(E),
        0,
        zero(E),
        zero(E),
        zero(flat),
        zero(E),
        false,
        false,
    )
    finalise = TraceFinalise(
        keep_going,
        redraw,
        RedrawSlot(game, false, settings.first_step),
        settings.corrector,
    )
    return redrawn_trace(
        guarded_trace(traced_start, nothing, settings, finalise, accept_corrected, opening),
        finalise,
        settings,
        accept_corrected,
        context,
    )
end

"""
    redrawn_trace(outcome, finalise, settings, accept_corrected, context)

The `TraceResult` of a trace segment that ended with `outcome`. When `redraw` swapped the game there, the trace
restarts on the new game from the last accepted point (`outcome.progress`), a root of the old game, as a new segment
counted on from its steps.
"""
function redrawn_trace(
    outcome::NamedTuple,
    finalise::TraceFinalise,
    settings::TraceSettings{E},
    accept_corrected::Any,
    context::NamedTuple,
)::TraceResult{E} where {E <: Real}
    (; progress) = outcome
    (; slot) = finalise
    if !isnothing(outcome.exception) || !slot.pending
        return reached_state(progress, outcome, slot.game, settings, context)
    end
    slot.pending = false
    if progress.steps >= settings.max_steps
        return reached_state(progress, outcome, slot.game, settings, context)
    end
    return redrawn_trace(
        with_earlier_accuracy(
            guarded_trace(
                [progress.gaps; progress.relative_precision],
                [progress.gap_tangent; progress.tangent_precision],
                settings,
                finalise,
                accept_corrected,
                progress,
            ),
            outcome.logit_accuracy,
        ),
        finalise,
        settings,
        accept_corrected,
        context,
    )
end

"""A segment's `outcome` whose `logit_accuracy` also spans the earlier segments' `earlier` accuracy."""
function with_earlier_accuracy(outcome::NamedTuple, earlier::Real)::NamedTuple
    return (; outcome..., logit_accuracy = max(outcome.logit_accuracy, earlier))
end

"""
    guarded_trace(start, old_tangent, settings, finalise, accept_corrected, opening)

`(; progress, step_length, accepted_step, exhausted, logit_accuracy, unresolved, exception)` of a PALC segment on `finalise.slot`'s
game from `start = [w; ρ]`, the point of `opening` (the `TraceProgress` before it), for the steps `settings.max_steps`
leaves: the progress after it (`branch_progress`), the step length its last corrector tried, the step of the last
accepted point (the slot's, carried from an earlier segment when this one accepted none), the corrector's confirmed stall
at rounding noise (or `nothing`), the largest last correction of an accepted point, whether its predictor fell below
the float type's resolution (`step_unresolved`), and `nothing` for a segment that
ran, else the third-party failure it ended on. From the flat start (`old_tangent` is `nothing`) the first tangent is
BifurcationKit's secant; a restart after a redraw opens along `restart_tangent`, oriented by `old_tangent`.
BifurcationKit throws a `String` when Newton misses its first point and an `ErrorException` at its second; a
non-finite Jacobian reaches LAPACK, and so does a singular restart tangent. Any other exception propagates.
"""
function guarded_trace(
    start::AbstractVector{E},
    old_tangent::Union{Nothing, AbstractVector{E}},
    settings::TraceSettings{E},
    finalise::TraceFinalise,
    accept_corrected::Any,
    opening::TraceProgress{E},
)::NamedTuple where {E <: Real}
    corrector = CorrectorReach(
        accept_corrected,
        settings.corrector,
        settings.geometry,
        similar(start),
        similar(start),
        zero(E),
    )
    return segment_outcome(
        segment_branch(
            trace_iterator(
                finalise.slot.game,
                start,
                settings,
                finalise,
                corrector,
                finalise.slot.step_length,
                settings.max_steps - opening.steps,
            ),
            finalise.slot.factorization,
            old_tangent,
            opening,
        ),
        corrector,
        finalise,
    )
end

"""
`guarded_trace`'s outcome of a segment that `ran` (`segment_branch`) under `corrector`, with the step of the last
accepted point, which `finalise` sets in its slot at each one.
"""
function segment_outcome(
    ran::NamedTuple,
    corrector::CorrectorReach{C, E},
    finalise::TraceFinalise,
)::NamedTuple where {C, E <: Real}
    return (;
        ran.progress,
        step_length = ifelse(isnothing(ran.exception), corrector.step_length, zero(E)),
        accepted_step = E(finalise.slot.step_length),
        exhausted = ifelse(
            isnothing(ran.exception) && noise_confirmed(corrector),
            corrector.noise.suspect,
            nothing,
        ),
        logit_accuracy = corrector.noise.accuracy,
        unresolved = isnothing(ran.exception) && corrector.unresolved,
        ran.exception,
    )
end

"""How a guarded BifurcationKit call ended other than with its result: `Thrown` or `Ended`."""
abstract type GuardedEnd end

"""An exception BifurcationKit threw that `expected_failure` names, held as a value."""
struct Thrown{X} <: GuardedEnd
    exception::X
end

"""BifurcationKit's `continuation!` ran to its end."""
struct Ended <: GuardedEnd end

"""
    segment_branch(iterator, factorization, old_tangent, opening)

`(; progress, exception)` of BifurcationKit's continuation of `iterator` from the point of `opening`: the progress
after its branch (`branch_progress`), and the exception it threw or `nothing`. A segment that throws before its
branch exists, or accepts no point, stays at `opening` with the restart tangent once that is computed. Only an
exception `expected_failure` names is caught (`Thrown`); any other propagates. Each step is its own method, as
BifurcationKit's results are not inferable.
"""
function segment_branch(
    iterator::BifurcationKit.ContIterable,
    factorization::Union{Nothing, LU{E}},
    old_tangent::Union{Nothing, AbstractVector{E}},
    opening::TraceProgress{E},
)::NamedTuple where {E <: Real}
    return restarted_segment(
        iterator,
        guarded_restart(iterator, factorization, old_tangent, opening),
        opening,
    )
end

restarted_segment(
    ::BifurcationKit.ContIterable,
    thrown::Thrown,
    opening::TraceProgress,
)::NamedTuple = (; progress = opening, thrown.exception)

function restarted_segment(
    iterator::BifurcationKit.ContIterable,
    restart::NamedTuple,
    opening::TraceProgress{E},
)::NamedTuple where {E <: Real}
    opened = TraceProgress{E}(
        opening.gaps,
        opening.relative_precision,
        opening.steps,
        opening.precision_min,
        opening.precision_max,
        restart.gap_tangent,
        restart.tangent_precision,
        opening.unmoved,
        opening.looped,
    )
    return opened_segment(iterator, guarded_state(iterator, restart.opening), opened)
end

opened_segment(
    ::BifurcationKit.ContIterable,
    thrown::Thrown,
    opened::TraceProgress,
)::NamedTuple = (; progress = opened, thrown.exception)

function opened_segment(
    iterator::BifurcationKit.ContIterable,
    state::BifurcationKit.ContState,
    opened::TraceProgress,
)::NamedTuple
    return continued_segment(
        iterator,
        state,
        BifurcationKit.ContResult(iterator, state),
        opened,
    )
end

"""The segment from `state`, its records saved in `branch` as BifurcationKit continues it."""
function continued_segment(
    iterator::BifurcationKit.ContIterable,
    state::BifurcationKit.ContState,
    branch::BifurcationKit.ContResult,
    opened::TraceProgress,
)::NamedTuple
    # `guarded_continuation!` runs before `ended_segment` reads the branch it filled.
    return ended_segment(guarded_continuation!(iterator, state, branch), branch, opened)
end

ended_segment(
    ::Ended,
    branch::BifurcationKit.ContResult,
    opened::TraceProgress,
)::NamedTuple = (; progress = branch_progress(branch.branch, opened), exception = nothing)

ended_segment(
    thrown::Thrown,
    branch::BifurcationKit.ContResult,
    opened::TraceProgress,
)::NamedTuple = (; progress = branch_progress(branch.branch, opened), thrown.exception)

"""`restart_opening`, or the `Thrown` failure that `expected_failure` names; any other exception propagates."""
function guarded_restart(
    iterator::BifurcationKit.ContIterable,
    factorization::Union{Nothing, LU{E}},
    old_tangent::Union{Nothing, AbstractVector{E}},
    opening::TraceProgress{E},
)::Union{NamedTuple, GuardedEnd} where {E <: Real}
    try
        return restart_opening(iterator, factorization, old_tangent, opening)
    catch exception
        return thrown_failure(exception)
    end
end

"""`opened_state`, or the `Thrown` failure that `expected_failure` names; any other exception propagates."""
function guarded_state(
    iterator::BifurcationKit.ContIterable,
    opening::Union{Nothing, AbstractVector},
)::Union{BifurcationKit.ContState, GuardedEnd}
    try
        return opened_state(iterator, opening)
    catch exception
        return thrown_failure(exception)
    end
end

"""BifurcationKit's `continuation!`: `Ended` once it ends, or the `Thrown` failure that `expected_failure` names."""
function guarded_continuation!(
    iterator::BifurcationKit.ContIterable,
    state::BifurcationKit.ContState,
    branch::BifurcationKit.ContResult,
)::GuardedEnd
    try
        BifurcationKit.continuation!(iterator, state, branch)
        return Ended()
    catch exception
        return thrown_failure(exception)
    end
end

"""The `Thrown` failure holding `exception` if `expected_failure` names it; any other exception is rethrown."""
function thrown_failure(exception::X)::Thrown{X} where {X}
    if !expected_failure(exception)
        throw(exception)
    end
    return Thrown(exception)
end

"""Whether `exception` is a failure BifurcationKit reports by throwing: an `uncorrected_start` or a `solver_blowup`."""
expected_failure(exception::Any)::Bool = uncorrected_start(exception) ||
    solver_blowup(exception)

"""
    restart_opening(iterator, factorization, old_tangent, opening)

The flat start (`old_tangent` is `nothing`) opens on BifurcationKit's secant with no point to aim at, and keeps
`opening`'s tangent until a point is accepted.
"""
function restart_opening(
    ::BifurcationKit.ContIterable,
    ::Nothing,
    ::Nothing,
    opening::TraceProgress,
)::NamedTuple
    return (; opening = nothing, opening.gap_tangent, opening.tangent_precision)
end

function restart_opening(
    iterator::BifurcationKit.ContIterable,
    factorization::LU{E},
    old_tangent::AbstractVector{E},
    ::TraceProgress{E},
)::NamedTuple where {E <: Real}
    return restart_opening(iterator, factorization, old_tangent)
end

function restart_opening(
    ::BifurcationKit.ContIterable,
    ::Nothing,
    ::AbstractVector,
    ::TraceProgress,
)::NamedTuple
    # The swapped game was non-finite at the swap point (`ShiftedGame`); LAPACK's own check reports it so.
    throw(ArgumentError("matrix contains Infs or NaNs"))
end

"""
    reached_state(progress, outcome, game, settings, context)

The `TraceResult` (`Ok(Traced)` or `Err(TraceFailure)`) of a trace whose `progress` holds the last accepted point and
whose `outcome` (`guarded_trace`) is the last corrector's step length, its `exhausted` record, or the BifurcationKit
exception that ended it. A corrector at rounding noise fails `PrecisionExhausted`, its context carrying the precision
`lambda` and the `correction` in logits there.
"""
function reached_state(
    progress::TraceProgress{E},
    outcome::NamedTuple,
    game::LogitGame,
    settings::TraceSettings{E},
    context::NamedTuple,
)::TraceResult{E} where {E <: Real}
    target_precision = game.target_precision
    if !isnothing(outcome.exhausted)
        (; relative_precision, correction) = outcome.exhausted
        @debug "trace stopped" _group = :trace reason = :precision_exhausted relative_precision correction
        return Err(TraceFailure(
            PrecisionExhausted,
            (; lambda = target_precision * relative_precision, correction, stopped_context(
                progress,
                outcome,
                target_precision,
                context,
            )...),
        ))
    end
    if isnothing(outcome.exception) && progress.relative_precision == one(E)
        return Ok(Traced(copy(progress.gaps), progress.steps, E(outcome.logit_accuracy)))
    end
    # Why the trace stopped short: a start that would not correct, a broken linear solve, a step below resolution, a point that did not move, a loop of the branch,
    # spent steps, or else `keep_going` or a refusal (`STOP_CAUSES`).
    reason = if uncorrected_start(outcome.exception)
        :uncorrected_start
    elseif !isnothing(outcome.exception)
        :solver_blowup
    elseif outcome.unresolved
        :unresolved_step
    elseif progress.unmoved
        :unmoved
    elseif progress.looped
        :loop
    elseif progress.steps >= settings.max_steps
        :max_steps
    else
        :stopped_short
    end
    @debug "trace stopped" _group = :trace reason
    return Err(TraceFailure(
        STOP_CAUSES[reason],
        stopped_context(progress, outcome, target_precision, context),
    ))
end

# The cause a trace that stops short for each reason fails with.
const STOP_CAUSES = (;
    uncorrected_start = TraceStuck, solver_blowup = Singular, max_steps = NoConvergence, unmoved = TraceStuck, unresolved_step = TraceStuck, loop = TraceStuck,
    stopped_short = TraceStuck,
)

"""
    stopped_context(progress, outcome, target_precision, context)

The context of a trace that stopped at `progress`'s last accepted point after `outcome`: `context`, the steps, the
precision and its range, the logits, the step held, the arclength step of the last accepted point (`accepted_step`,
zero when no step was accepted), the tangent in logits, the `logit_accuracy` of its accepted points and the exception.
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
        progress.precision_min, progress.precision_max, progress.unmoved, progress.looped,
        logits = progress.relative_precision .* progress.gaps, progress.relative_precision,
        step_length = E(outcome.step_length),
        accepted_step = ifelse(progress.steps > 0, E(outcome.accepted_step), zero(E)),
        logit_tangent = progress.relative_precision .* progress.gap_tangent .+
            progress.tangent_precision .* progress.gaps,
        progress.tangent_precision,
        logit_accuracy = E(outcome.logit_accuracy),
        outcome.exception,
    )
end
