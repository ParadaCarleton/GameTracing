using BifurcationKit: BifurcationKit
using LinearAlgebra: dot, norm
using Logging: @debug
"""
    CorrectorSettings{E}(; reach, max_kl, logit_tolerance, pin_tolerance, contraction_limit,
                         contraction_offset, newton_iterations, residual_tolerance)

The corrector's controls (`CorrectorReach`); `nothing` switches a limit off. `reach`: a corrector that moves a logit
further from the predictor than this times the predictor's own largest logit move is refused. `shift_share`: the share
of that reach a redraw's predicted root shift may use (`RootShift`). `max_kl`: a corrected point whose largest per-logit probability KL from
the predictor passes this is refused. A step's turn is limited by `StepControl`'s `turn_limit`.

Convergence is scale-free, as in Gambit's tracer (path.cc:372, `c_tol` in log probabilities): a Newton correction that
moves no logit more than `logit_tolerance`, at a relative precision within the relative `pin_tolerance` of its pin.
A correction that moves the logits more than `contraction_limit` times the one before it (plus `logit_tolerance`
times `contraction_offset`) fails to contract; BifurcationKit's step growth reads the same rate (`corrector_quality`).
`newton_iterations`: Newton iterations per corrector (BifurcationKit's default, 25; Gambit's is 100).
`residual_tolerance`: BifurcationKit's residual tolerance `tol`, which a corrector must also meet. The branch residual
`w - P G` is in logits at the target precision `P`, so its size grows with `P`; the default `Inf` leaves convergence to
the logit test.
"""
Base.@kwdef struct CorrectorSettings{E <: Real}
    reach::Union{Nothing, E} = 2.0f0
    # Measured shares at the swap fixtures (test/games.jl): NEAR 0.075; PITCHFORK 4.0, FAR 77, FOLD 1057.
    # Half the reach leaves the corrector the other half for the branch's curvature, and splits 0.075 and 4 evenly.
    shift_share::E = 0.5f0
    max_kl::Union{Nothing, E} = nothing
    logit_tolerance::E = 1.0f-4
    pin_tolerance::E = 1.0f-5
    contraction_limit::Union{Nothing, E} = 0.6f0
    contraction_offset::E = 0.1f0
    newton_iterations::Int = 25
    residual_tolerance::E = Inf32
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
    suspect::Union{
        Nothing,
        @NamedTuple{relative_precision::E, correction::E, confirmed::Bool},
    }
end

"""
    CorrectorState(step_length, predictor_move, correction, stalls, settled_here, noise)

The scalars a `CorrectorReach` carries between BifurcationKit's calls: the current step's `step_length` and its
predictor's largest logit change `predictor_move`; the current solve's last `correction` in logits, its
non-contracting corrections in a row (`stalls`) and whether it settled (`settled_here`); and the trace's `noise`
(`NoiseRecord`).
"""
struct CorrectorState{E <: Real}
    step_length::E
    predictor_move::E
    correction::E
    stalls::Int
    settled_here::Bool
    noise::NoiseRecord{E}
end

"""`state` with the given fields replaced."""
function restated(
        state::CorrectorState{E};
        step_length::Real = state.step_length,
        predictor_move::Real = state.predictor_move,
        correction::Real = state.correction,
        stalls::Integer = state.stalls,
        settled_here::Bool = state.settled_here,
        noise::NoiseRecord{E} = state.noise,
    )::CorrectorState{E} where {E <: Real}
    return CorrectorState{E}(step_length, predictor_move, correction, stalls, settled_here, noise)
end

"""
    CorrectorReach(accept_corrected, settings, geometry, previous, predicted, step_length)

The trace's BifurcationKit `callback_newton`, which also tests the corrector's convergence: it marks a `settled`
iterate (`settled_here`), which ends Newton's method converged through `CorrectorSettled`, and a correction that fails
to contract stops it (`corrected!`).
Two such corrections in a row, each larger than the predictor's own move in logits, are suspected rounding noise
(`noise`), confirmed at the next halved step; the check then refuses every corrector (`noise_confirmed`).
A non-finite residual stops the corrector,
and so does any iterate whose logits are further from the predictor's than `settings.reach` times the predictor's own
largest logit move (`within_reach`); the step then halves. On a smooth branch that distance is the branch's curvature
times `ds² / 2`, so halving brings any step under it, while an iterate that lands on another branch is far from it. A converged closing point is refused the same way past `reach`, past a largest probability KL
`KL(corrected ‖ predicted)` of `settings.max_kl` (`largest_kl`). A predictor that did not move has no reach to compare.
`accept_corrected(logits, ρ, converged)` then sees every other closing point. `previous` and `predicted` hold the
current step's start and predictor, `iterate` the current solve's last iterate, all updated in place; `state`
(`CorrectorState`) holds the scalars that change between calls.
"""
struct CorrectorReach{C, E <: Real}
    accept_corrected::C
    settings::CorrectorSettings{E}
    geometry::TraceGeometry{E}
    previous::Vector{E}
    predicted::Vector{E}
    iterate::Vector{E}
    state::Base.RefValue{CorrectorState{E}}  # lint: ref # lint: non-public-access  by design: callback_newton returns a verdict, and its state between calls is one immutable value here (RefValue is a Ref's concrete type)
end

"""`check`'s `CorrectorState` after replacing the fields named in `changes`."""
function restate!(check::CorrectorReach{C, E}; changes...)::CorrectorState{E} where {C, E <: Real}
    check.state[] = restated(check.state[]; changes...)
    return check.state[]
end

"""Whether `check`'s corrector met rounding noise, confirmed at a halved step (`corrected!`)."""
function noise_confirmed(check::CorrectorReach)::Bool
    return !isnothing(check.state[].noise.suspect) && check.state[].noise.suspect.confirmed
end

"""
    CorrectorSettled(check)

BifurcationKit's `convergence_test` (`NewtonPar`) of the trace: whether the `CorrectorReach` `check` settled the
current iterate. BifurcationKit calls it after the callback `check` on each iterate.
"""
struct CorrectorSettled{R <: CorrectorReach}
    check::R
end

# fatou-ignore unused-argument: BifurcationKit passes the iterate and keywords, which `check` has already read
(test::CorrectorSettled)(::NamedTuple; keywords...)::Bool = test.check.state[].settled_here

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
        accept_corrected,
        settings,
        geometry,
        collect(previous),
        collect(predicted),
        collect(previous),
        Ref(CorrectorState{E}(step_length, E(NaN32), E(NaN32), 0, false, NoiseRecord(zero(E), nothing))),  # lint: ref  see `state`
    )
end

"""Whether `check` accepts its corrector's converged closing point `corrected` (reach and KL)."""
function closing_accepted(
        check::CorrectorReach{C, E},
        corrected::AbstractVector{E},
    )::Bool where {C, E <: Real}
    (; settings) = check
    if !within_reach(check, corrected)
        @debug "corrector refused" _group = :trace reason = :reach predictor_move = check.state[].predictor_move
        return false
    end
    if !within_limit(settings.max_kl, largest_kl(corrected, check.predicted))
        @debug "corrector refused" _group = :trace reason = :kl
        return false
    end
    return true
end

function (check::CorrectorReach{C, E})(
        iterate::I;
        fromNewton::Bool = true,
        # fatou-ignore unused-argument: BifurcationKit passes keyword arguments this callback ignores
        keywords...,
    )::Bool where {C, E <: Real, I <: NamedTuple}
    if fromNewton
        # Newton's closing call is its only one without `J`.
        if !haskey(iterate, :J)
            return isfinite(iterate.residual)
        end
        if iterate.step == 0
            restate!(check; predictor_move = E(NaN32))
        end
        return corrected!(check, iterate, isfinite(iterate.residual))
    end
    # PALC's closing call, marked `closing`, ends every corrector.
    if !haskey(iterate, :closing)
        @debug "corrector iterate" _group = :trace step = iterate.step residual = iterate.residual relative_precision = last(
            iterate.x,
        )
        if iterate.step == 0
            copyto!(check.predicted, iterate.x)
            copyto!(check.previous, iterate.z0.u)
            restate!(
                check;
                step_length = arclength_distance(iterate.z0.u, iterate.x, check.geometry),
                predictor_move = logit_change(iterate.z0.u, iterate.x),
            )
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
    # An exact root converges unsettled on a Newton update of exactly zero (`unmoved`), so it meets the same tests.
    converged = isfinite(iterate.residual) && (iterate.converged || get(iterate, :unmoved, false))
    @debug "corrector closed" _group = :trace step = iterate.step residual = iterate.residual converged step_length = check.state[].step_length
    if converged && !closing_accepted(check, iterate.x)
        return false
    end
    if !check.accept_corrected(logits_at(iterate.x), iterate.p, converged)
        return false
    end
    # A solve that settled at its start has no correction (NaN). An unsettled exact root's next correction is 0.
    closing_correction = ifelse(check.state[].settled_here, check.state[].correction, zero(E))
    if converged && !isnan(closing_correction)
        restate!(
            check;
            noise = NoiseRecord(max(check.state[].noise.accuracy, closing_correction), check.state[].noise.suspect),
        )
    end
    return true
end

"""
    corrected!(check, iterate, proceed)

`check`'s verdict on a Newton `iterate` that its other tests let `proceed`, in logits `z = ρ w`. The first iterate
is the solve's start and never settles. Each later one is a correction `Δz` from the iterate before: it settles
(`settled_here`, read by `CorrectorSettled`) when `settled`. A correction past `contraction_limit` times the one before (plus
`logit_tolerance · contraction_offset`) fails to contract, Gambit's test (path.cc), and stops the corrector, so the
step halves. Two in a row, each larger than the predictor's own move, are suspected noise (`noise`); the
step still halves. A second such solve before any settles, its correction at least half the suspect's, confirms it
(`noise_confirmed`): halving shrinks the corrections of a predictor past a fold, not noise. Every later solve refuses.
"""
function corrected!(check::CorrectorReach{C, E}, iterate::NamedTuple, proceed::Bool)::Bool where {C, E <: Real}
    if noise_confirmed(check)
        return false
    end
    if iterate.step == 0
        copyto!(check.iterate, iterate.x)
        restate!(check; correction = E(NaN32), stalls = 0, settled_here = false)
        return proceed
    end
    correction, before = logit_change(check.iterate, iterate.x), check.state[].correction
    copyto!(check.iterate, iterate.x)
    restate!(check; correction)
    if !proceed
        return false
    end
    if settled(check, iterate, correction)
        restate!(check; settled_here = true, noise = NoiseRecord(check.state[].noise.accuracy, nothing))
        return true
    end
    (; contraction_limit, contraction_offset, logit_tolerance) = check.settings
    # The first correction has no rate (`before` is NaN), so it contracts.
    if isnan(before) ||
            within_limit(
            scaled_limit(
                contraction_limit,
                before + logit_tolerance * contraction_offset,
            ),
            correction,
        )
        restate!(check; stalls = 0)
        return true
    end
    if !(check.state[].predictor_move <= correction)
        @debug "corrector stopped" _group = :trace reason = :contraction correction before
        return false
    end
    if restate!(check; stalls = check.state[].stalls + 1).stalls < 2
        return true
    end
    # Halving leaves noise's corrections alike (ratio 1) and shrinks an overshot predictor's by 1/4; 1/2 splits them.
    suspect = check.state[].noise.suspect
    confirmed = !isnothing(suspect) && correction >= suspect.correction / 2
    @debug "corrector stopped" _group = :trace reason = :noise confirmed correction predictor_move = check.state[].predictor_move
    restate!(
        check;
        noise = NoiseRecord(check.state[].noise.accuracy, (; relative_precision = last(iterate.x), correction, confirmed)),
    )
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
function settled(check::CorrectorReach, iterate::I, correction::Real)::Bool where {
        I <:
        NamedTuple,
    }
    relative_precision = last(iterate.x)
    offset = pin_residual(iterate)
    pin_scale = max(abs(relative_precision), abs(relative_precision - offset))
    return correction <= check.settings.logit_tolerance &&
        abs(offset) <= check.settings.pin_tolerance * pin_scale
end

"""
Whether `point` moved no logit further from the predictor's than `check.settings.reach` times the predictor's own
largest logit move (`logit_change`). Logits are target-free, as the convergence test is; a weighted norm can shrink a
branch jump (frozen Fisher weights) or swell a settled point (gaps `w = P G` at a far target).
"""
function within_reach(check::CorrectorReach, point::AbstractVector)::Bool
    if !(check.state[].predictor_move > 0)
        return true
    end
    return within_limit(
        scaled_limit(check.settings.reach, check.state[].predictor_move),
        logit_change(check.predicted, point),
    )
end

"""
    TraceProgress(gaps, relative_precision, steps, precision_min, precision_max, gap_tangent, tangent_precision, unmoved,
        looped)

The last accepted point `(gaps, relative_precision)` of a trace, its step count, the range of the relative precision
over accepted points and the tangent there (both in payoff gaps), read from BifurcationKit's branch (`branch_progress`).
`unmoved` marks a trace that stopped on an accepted point bitwise equal to the one before it (`TraceFinalise`), `looped` one that closed a loop of the branch (`looped`).
"""
struct TraceProgress{E <: Real}
    gaps::Vector{E}
    relative_precision::E
    steps::Int
    precision_min::E
    precision_max::E
    gap_tangent::Vector{E}
    tangent_precision::E
    unmoved::Bool
    looped::Bool
end

"""
BifurcationKit's branch record of the trace's `point = [w; ρ]` and the state's tangent: the gaps `w`, their tangent
and the tangent's precision part. `branch_progress` reads the trace's progress back from these records.
"""
function trace_record_point(
        point::AbstractVector{E}, ::Real;
        state::Any,
        # fatou-ignore unused-argument: BifurcationKit passes keyword arguments this record ignores
        keywords...,
    )::NamedTuple where {E <: Real}
    return (;
        gaps = point[begin:(end - 1)], gap_tangent = state.τ.u[
            begin:(end - 1),
        ], tangent_precision = state.τ.p,
    )
end

"""A branch record's relative precision."""
function record_precision(record::R)::fieldtype(R, :param) where {R <: NamedTuple}
    return record.param
end

"""Whether two branch records hold the same point `(w, ρ)`, bitwise: a step below the point's rounding."""
same_point(record::NamedTuple, other::NamedTuple)::Bool = record.param == other.param &&
    record.gaps == other.gaps

"""
    tangent_turn(left, right)

The angle in radians between two branch records' tangents `[gap_tangent; tangent_precision]`, in the Euclidean product;
zero when either is zero.
"""
function tangent_turn(left::R, right::R)::fieldtype(R, :param) where {R <: NamedTuple}
    lengths = sqrt(
        (dot(left.gap_tangent, left.gap_tangent) + left.tangent_precision^2) *
            (dot(right.gap_tangent, right.gap_tangent) + right.tangent_precision^2),
    )
    if lengths == 0
        return zero(lengths)
    end
    cosine = (dot(left.gap_tangent, right.gap_tangent) + left.tangent_precision * right.tangent_precision) / lengths
    return acos(clamp(cosine, -one(cosine), one(cosine)))
end

"""
    looped(history, current)

Whether `current`, a point with `gaps` and `param` that follows the branch records `history`, closes a loop of the
branch (an isola). Gate: the tangents of `history` have turned through at least 2π in total (`tangent_turn`), as a closed
curve must (Fenchel's theorem). Then Brent's cycle detection: the anchor is the record at the largest power-of-two index
not after the last record (indices from 1), and the trace has looped when `current` lies closer to the anchor than the
last step is long, with the anchor at least two records before `current`. Distances are plain Euclidean on
`[gaps; param]`, as `TraceFinalise` holds no `TraceGeometry`.
"""
function looped(history::AbstractVector{R}, current::NamedTuple)::Bool where {R <: NamedTuple}
    anchor_index = prevpow(2, max(length(history), 1))
    if anchor_index > length(history) - 1
        return false
    end
    anchor = history[firstindex(history) + anchor_index - 1]
    point = [current.gaps; current.param]
    step_length = norm(point - [last(history).gaps; last(history).param])
    if !(norm(point - [anchor.gaps; anchor.param]) < step_length)
        return false
    end
    turn = sum(tangent_turn.(view(history, firstindex(history):(lastindex(history) - 1)), view(history, (firstindex(history) + 1):lastindex(history))))
    return turn >= 2 * oftype(turn, pi)
end

"""
    branch_progress(branch, opening)

The `TraceProgress` after a segment whose BifurcationKit `branch` of `trace_record_point` records starts at the point of
`opening`, the progress before the segment: its last record, its steps counted on from `opening.steps`, the precision
range of both, `unmoved` when its last two records hold the same point and `looped` when its last record closes a loop (`looped`). A branch of its start alone accepted no
point and leaves `opening` as it was, tangent included.
"""
function branch_progress(branch::AbstractVector, opening::TraceProgress{E})::TraceProgress{
        E,
    } where {E <: Real}
    if length(branch) == 1
        return opening
    end
    record = last(branch)
    unmoved = same_point(record, branch[end - 1])
    closed = looped(view(branch, firstindex(branch):(lastindex(branch) - 1)), record)
    return TraceProgress{E}(
        copy(record.gaps),
        record.param,
        opening.steps + record.step,
        min(opening.precision_min, minimum(record_precision, branch)),
        max(opening.precision_max, maximum(record_precision, branch)),
        copy(record.gap_tangent),
        record.tangent_precision,
        unmoved,
        closed,
    )
end

"""
    TraceFinalise(keep_going, redraw, slot, settings)

BifurcationKit's `finalise_solution` of one trace segment. An accepted point bitwise equal to the branch's last one (a
step below the point's rounding) stops the trace, a stall, as does a point that closes a loop of the branch (`looped`). The trace goes on while `keep_going(logits,
relative_precision)` holds. Then `redraw(logits, relative_precision, shift)` may return a new game, where `shift` is
the point's `RootShift` check under `settings`: `slot`, a `Ref` to the current `RedrawSlot` (the constructor
wraps the one it is given), takes it and the
segment stops, for `trace_endpoint` to correct the point onto it and restart. A `redraw` that returns `nothing` or the
current game swaps nothing. `JULIA_DEBUG=trace` lists every point.
"""
struct TraceFinalise{A, D, S, C <: CorrectorSettings}
    keep_going::A
    redraw::D
    slot::Base.RefValue{S}  # lint: ref # lint: non-public-access  by design: finalise_solution returns a Bool, and a redraw's swap reaches the trace as one immutable value here (RefValue is a Ref's concrete type)
    settings::C
end

function TraceFinalise(keep_going::A, redraw::D, slot::S, settings::C)::TraceFinalise{A, D, S, C} where {A, D, S, C <: CorrectorSettings}
    return TraceFinalise{A, D, S, C}(keep_going, redraw, Ref(slot), settings)  # lint: ref  see `slot`
end

function (finalise::TraceFinalise)(
        point::P, tangent::T, step::S, result::Any;
        state::Any,
        iter::Any,
        # fatou-ignore unused-argument: BifurcationKit passes keyword arguments this callback ignores
        keywords...,
    )::Bool where {P, T, S <: Integer}
    @debug "trace point" _group = :trace step relative_precision = point.p state = copy(
        point.u,
    ) tangent = copy(tangent.u)
    # BifurcationKit saves the point after this call, so the branch's last record is the point before it.
    if same_point(
            (;
                param = point.p, gaps = view(
                    point.u,
                    firstindex(point.u):(lastindex(point.u) - 1),
                ),
            ),
            last(result.branch),
        )
        return false
    end
    current = (; param = point.p, gaps = view(point.u, firstindex(point.u):(lastindex(point.u) - 1)))
    if looped(result.branch, current)
        return false
    end
    logits = logits_at(point.u)
    if !finalise.keep_going(logits, point.p)
        return false
    end
    slot = restated(finalise.slot[]; step_length = abs(state.ds))
    finalise.slot[] = slot
    # A redraw swaps the map for the steps after this point, and the endpoint has none.
    if point.p == 1
        return true
    end
    algorithm = iter.alg
    shift = RootShift(
        point.u,
        state.ds .* tangent.u,
        palc_border(algorithm.dotθ, eltype(point.u)(algorithm.θ), tangent.u),
        finalise.settings,
    )
    # The slot is pending only after a swap here: a restart clears it before the segment runs.
    finalise.slot[] = swapped(slot, shift, finalise.redraw(logits, point.p, shift))
    return !finalise.slot[].pending
end
