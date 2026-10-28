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
    # BifurcationKit's constructors are not inferable, so each is built in its own function and passed on unnamed.
    return crossing_walk(
        homotopy_iterable(homotopy_problem(homotopy, point), algorithm, options, callback_newton),
        HomotopyJacobian(homotopy),
        point,
    )
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

"""BifurcationKit's continuation iterator of `problem` under `algorithm` and `options`, with `callback_newton`."""
function homotopy_iterable(
    problem::BifurcationKit.AbstractBifurcationProblem,
    algorithm::BifurcationKit.AbstractContinuationAlgorithm,
    options::BifurcationKit.ContinuationPar,
    callback_newton::Any,
)::BifurcationKit.ContIterable
    return BifurcationKit.ContIterable(
        problem,
        BifurcationKit.update(algorithm, options, nothing),
        options;
        normC = BifurcationKit.norminf,
        verbosity = 0,
        callback_newton,
    )
end

"""
    crossing_walk(iterable, jacobian, point)

`homotopy_crossing`'s result: BifurcationKit's `iterable` run from `point` at `s = 0` (`crossing_steps`).
"""
function crossing_walk(iterable::BifurcationKit.ContIterable, jacobian::HomotopyJacobian, point::AbstractVector{E})::CrossingResult{E} where {E <: Real}
    return crossing_steps(iterable, jacobian, sign(det(jacobian(point, zero(E)))), first(something(iterate(iterable))))
end

"""
    crossing_steps(iterable, jacobian, start_sign, start)

The `CrossingResult` of `iterable`'s states from its first, `start`: each accepted point is checked for a turn in
`s` (the tangent's `s` part falls below zero) and for `det jacobian` leaving `start_sign`. The tangent is not
oriented by the bordered determinant, so it turns only at a fold.
"""
function crossing_steps(
    iterable::BifurcationKit.ContIterable,
    jacobian::HomotopyJacobian,
    start_sign::Real,
    start::S,
)::CrossingResult{eltype(BifurcationKit.getx(start))} where {S}
    state = start
    # While `done` holds, `iterate` returns the next state, converged at a new point or not.
    while BifurcationKit.done(iterable, state)
        state = first(something(iterate(iterable, state)))::S
        if BifurcationKit.converged(state)
            @debug "crossing point" _group = :homotopy_crossing blend = BifurcationKit.getp(state)
            if BifurcationKit.gettangent(state).p < 0
                return CrossingResult(copy(BifurcationKit.getx(state)), :fold, state.step)
            end
            if sign(det(jacobian(BifurcationKit.getx(state), BifurcationKit.getp(state)))) != start_sign
                return CrossingResult(copy(BifurcationKit.getx(state)), :crossing, state.step)
            end
        end
    end
    reached = BifurcationKit.converged(state) && BifurcationKit.getp(state) == 1
    return CrossingResult(copy(BifurcationKit.getx(state)), ifelse(reached, :crossed, :stopped), state.step)
end
