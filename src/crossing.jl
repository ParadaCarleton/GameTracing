using BifurcationKit: BifurcationKit
using LinearAlgebra: det
using Logging: @debug

"""
    HomotopyMap(start_problem, end_problem)

The convex homotopy `H(u, s) = (1 - s) F₀(u) + s F₁(u)` between two BifurcationKit problems' vector fields, each at
its own parameters. `s` is the homotopy's continuation parameter; neither problem's own parameter moves.
"""
struct HomotopyMap{A <: BifurcationKit.AbstractBifurcationProblem, B <: BifurcationKit.AbstractBifurcationProblem}
    start_problem::A
    end_problem::B
end

"""`problem`'s vector field at `state`, at the problem's own parameters."""
function own_residual(problem::BifurcationKit.AbstractBifurcationProblem, state::AbstractVector{E})::AbstractVector{E} where {E <: Real}
    return BifurcationKit.residual(problem, state, BifurcationKit.getparams(problem))
end

"""`problem`'s Jacobian in the state at `state`, at the problem's own parameters."""
function own_jacobian(problem::BifurcationKit.AbstractBifurcationProblem, state::AbstractVector{E})::AbstractMatrix{E} where {E <: Real}
    return BifurcationKit.jacobian(problem, state, BifurcationKit.getparams(problem))
end

"""`H(u, s)` of a `HomotopyMap`, as BifurcationKit's vector field."""
struct HomotopyResidual{H <: HomotopyMap}
    homotopy::H
end

function (field::HomotopyResidual)(state::AbstractVector{E}, blend::Real)::AbstractVector{E} where {E <: Real}
    (; start_problem, end_problem) = field.homotopy
    return (1 - blend) .* own_residual(start_problem, state) .+ blend .* own_residual(end_problem, state)
end

"""`∂H/∂u` of a `HomotopyMap`, as BifurcationKit's Jacobian."""
struct HomotopyJacobian{H <: HomotopyMap}
    homotopy::H
end

function (jacobian::HomotopyJacobian)(state::AbstractVector{E}, blend::Real)::AbstractMatrix{E} where {E <: Real}
    (; start_problem, end_problem) = jacobian.homotopy
    return (1 - blend) .* own_jacobian(start_problem, state) .+ blend .* own_jacobian(end_problem, state)
end

"""`∂H/∂s = F₁ - F₀` of a `HomotopyMap`, exactly, as BifurcationKit's jet `R01`."""
struct HomotopySlope{H <: HomotopyMap}
    homotopy::H
end

function (slope::HomotopySlope)(state::AbstractVector{E}, ::Real)::AbstractVector{E} where {E <: Real}
    (; start_problem, end_problem) = slope.homotopy
    return own_residual(end_problem, state) .- own_residual(start_problem, state)
end

"""BifurcationKit's branch record of the homotopy: its parameter `s` alone."""
function homotopy_record(::Any, blend::L;
    # fatou-ignore unused-argument: BifurcationKit passes keyword arguments this callback ignores
    keywords...,
)::L where {L <: Real}
    return blend
end

"""
    CrossingResult(point, reason, steps)

A homotopy crossing's last accepted point, why it ended (`:crossed` at `s = 1`, else the failure) and its accepted steps.
"""
struct CrossingResult{E <: Real}
    point::Vector{E}
    reason::Symbol
    steps::Int
end

"""
    homotopy_crossing(start_problem, end_problem, point, algorithm, options; callback_newton)

A `CrossingResult`: `point`, a zero of `start_problem`'s vector field `F₀`, continued by BifurcationKit's
`algorithm` under `options` along `H(u, s) = (1 - s) F₀(u) + s F₁(u)` from `s = 0` to `s = 1`, where it is a zero of
`end_problem`'s `F₁`. `options` must bound `s` to `[0, 1]` (`p_min`, `p_max`); `callback_newton` is BifurcationKit's.

`reason` is `:crossed` when the path reaches `s = 1` with `s` rising at every accepted point and `det ∂H/∂u` of one
sign throughout. Otherwise it names the first failure: `:fold` (`s` turned back, so the zero at `s = 0` does not
persist to `s = 1`), `:crossing` (the determinant changed sign with `s` still rising: a singular point, as where the
path meets another branch), or `:stopped` (the corrector or the step budget ran out short of `s = 1`). `point` is the
last accepted point and `steps` the accepted steps. An even number of sign changes in one step goes unseen.
The Jacobians must be dense, for the determinant.
"""
function homotopy_crossing(
    start_problem::BifurcationKit.AbstractBifurcationProblem,
    end_problem::BifurcationKit.AbstractBifurcationProblem,
    point::AbstractVector{E},
    algorithm::BifurcationKit.AbstractContinuationAlgorithm,
    options::BifurcationKit.ContinuationPar;
    callback_newton::Any = BifurcationKit.cb_default,
)::CrossingResult{E} where {E <: Real}
    homotopy = HomotopyMap(start_problem, end_problem)
    watch = CrossingWatch(HomotopyJacobian(homotopy), sign(det(HomotopyJacobian(homotopy)(point, zero(E)))), copy(point), 0, :stopped)
    # BifurcationKit's constructors are not inferable, so each is built in its own function and passed on unnamed.
    BifurcationKit.continuation(
        homotopy_problem(homotopy, point),
        algorithm,
        options;
        normC = BifurcationKit.norminf,
        verbosity = 0,
        finalise_solution = watch,
        callback_newton,
    )
    return CrossingResult(watch.point, watch.reason, watch.steps)
end

"""BifurcationKit's problem for `homotopy` from `point` at `s = 0`, with its Jacobian and exact `∂H/∂s`."""
function homotopy_problem(homotopy::HomotopyMap, point::AbstractVector{E})::BifurcationKit.AbstractBifurcationProblem where {E <: Real}
    return BifurcationKit.BifurcationProblem(
        HomotopyResidual(homotopy),
        point,
        zero(E);
        J = HomotopyJacobian(homotopy),
        R01 = HomotopySlope(homotopy),
        inplace = false,
        record_from_solution = homotopy_record,
    )
end

"""
    CrossingWatch(jacobian, start_sign, point, steps, reason)

BifurcationKit's `finalise_solution` of a homotopy crossing: at each accepted point it stops the run with `reason`
`:fold` when the tangent turns back in `s`, `:crossing` when `det jacobian` leaves `start_sign`, else records the
point and its step count, `:crossed` once `s = 1`. A run that ends otherwise keeps `:stopped`.
"""
mutable struct CrossingWatch{J <: HomotopyJacobian, E <: Real}
    # Mutable because BifurcationKit's finalise callback returns only a `Bool`, as for `TraceProgress`.
    jacobian::J
    start_sign::E
    point::Vector{E}
    steps::Int
    reason::Symbol
end

function (watch::CrossingWatch)(point::P, tangent::T, step::S, ::Any;
    # fatou-ignore unused-argument: BifurcationKit passes keyword arguments this callback ignores
    keywords...,
)::Bool where {P, T, S <: Integer}
    @debug "crossing point" _group = :homotopy_crossing blend = point.p
    copyto!(watch.point, point.u)
    watch.steps = step
    if tangent.p < 0
        watch.reason = :fold
        return false
    end
    if sign(det(watch.jacobian(point.u, point.p))) != watch.start_sign
        watch.reason = :crossing
        return false
    end
    if point.p == one(point.p)
        watch.reason = :crossed
    end
    return true
end
