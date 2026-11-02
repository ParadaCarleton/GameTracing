using BifurcationKit: BifurcationKit
using LinearAlgebra: det
using Logging: @debug

"""
    HomotopyPasses(state, start_residual, end_residual, start_jacobian, end_jacobian)

The vector fields `F₀`, `F₁` and Jacobians `J₀`, `J₁` of a homotopy's two problems at `state`, the last state any
of them was asked for. Neither field depends on `s`, so the residual, `∂H/∂u` and `∂H/∂s` at one state share one
fused evaluation of each problem (`homotopy_parts!`).
"""
struct HomotopyPasses{E <: Real}
    state::Vector{E}
    start_residual::Vector{E}
    end_residual::Vector{E}
    start_jacobian::Matrix{E}
    end_jacobian::Matrix{E}
end

"""
    HomotopyMap(start_problem, end_problem, passes)

The convex homotopy `H(u, s) = (1 - s) F₀(u) + s F₁(u)` between two BifurcationKit problems' vector fields, each at
its own parameters, with the fields and Jacobians at the last state asked for (`HomotopyPasses`). `s` is the
homotopy's continuation parameter; neither problem's own parameter moves.
"""
struct HomotopyMap{A <: BifurcationKit.AbstractBifurcationProblem, B <: BifurcationKit.AbstractBifurcationProblem, P <: HomotopyPasses}
    start_problem::A
    end_problem::B
    passes::P
end

"""The `HomotopyMap` between `start_problem` and `end_problem`, its passes filled at `point`."""
function HomotopyMap(
    start_problem::BifurcationKit.AbstractBifurcationProblem,
    end_problem::BifurcationKit.AbstractBifurcationProblem,
    point::AbstractVector{E},
)::HomotopyMap where {E <: Real}
    start_residual, start_jacobian = own_parts(start_problem, point)
    end_residual, end_jacobian = own_parts(end_problem, point)
    return HomotopyMap(
        start_problem,
        end_problem,
        HomotopyPasses(collect(point), collect(start_residual), collect(end_residual), collect(start_jacobian), collect(end_jacobian)),
    )
end

"""`problem`'s vector field and Jacobian in the state at `state`, at the problem's own parameters, from one fused call."""
function own_parts(problem::BifurcationKit.AbstractBifurcationProblem, state::AbstractVector{E})::Tuple{AbstractVector{E}, AbstractMatrix{E}} where {E <: Real}
    return BifurcationKit.residual_and_jacobian(problem, state, BifurcationKit.getparams(problem))
end

"""
`homotopy`'s `HomotopyPasses`, evaluated at `state` unless they already hold it. The problems' fused calls return their
own workspaces, which the next call overwrites, so the passes keep copies.
"""
function homotopy_parts!(homotopy::HomotopyMap, state::AbstractVector{E})::HomotopyPasses{E} where {E <: Real}
    passes = homotopy.passes
    if passes.state != state
        start_residual, start_jacobian = own_parts(homotopy.start_problem, state)
        end_residual, end_jacobian = own_parts(homotopy.end_problem, state)
        copyto!(passes.state, state)
        copyto!(passes.start_residual, start_residual)
        copyto!(passes.end_residual, end_residual)
        copyto!(passes.start_jacobian, start_jacobian)
        copyto!(passes.end_jacobian, end_jacobian)
    end
    return passes
end

"""`H(u, s)` of a `HomotopyMap`, as BifurcationKit's vector field."""
struct HomotopyResidual{H <: HomotopyMap}
    homotopy::H
end

function (field::HomotopyResidual)(state::AbstractVector{E}, blend::Real)::AbstractVector{E} where {E <: Real}
    passes = homotopy_parts!(field.homotopy, state)
    return (1 - blend) .* passes.start_residual .+ blend .* passes.end_residual
end

"""`∂H/∂u` of a `HomotopyMap`, as BifurcationKit's Jacobian."""
struct HomotopyJacobian{H <: HomotopyMap}
    homotopy::H
end

function (jacobian::HomotopyJacobian)(state::AbstractVector{E}, blend::Real)::AbstractMatrix{E} where {E <: Real}
    passes = homotopy_parts!(jacobian.homotopy, state)
    return (1 - blend) .* passes.start_jacobian .+ blend .* passes.end_jacobian
end

"""`∂H/∂s = F₁ - F₀` of a `HomotopyMap`, exactly, as BifurcationKit's jet `R01`."""
struct HomotopySlope{H <: HomotopyMap}
    homotopy::H
end

function (slope::HomotopySlope)(state::AbstractVector{E}, ::Real)::AbstractVector{E} where {E <: Real}
    passes = homotopy_parts!(slope.homotopy, state)
    return passes.end_residual .- passes.start_residual
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
    homotopy = HomotopyMap(start_problem, end_problem, point)
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
