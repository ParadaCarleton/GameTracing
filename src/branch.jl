"""
    LogitGame(pass, target_precision)

A logit game whose equilibria solve `z = T(z, precision)` at precision `1 / temperature`. `pass(z, precision)` returns
`(; response, logit_slope, precision_slope)`: the map's value `T(z, precision)`, its Jacobian `∂T/∂z` and its slope
`∂T/∂precision`, from one evaluation. `T(z, 0) = 0`, so the flat equilibrium `z = 0` is the only root at zero precision.
`target_precision` is the precision a trace ends at.
"""
struct LogitGame{P, T <: Real}
    pass::P
    target_precision::T
end

"""The precision at `relative_precision`: the fraction of `game.target_precision` the trace has reached."""
function precision_at(game::LogitGame, relative_precision::Real)
    return relative_precision * game.target_precision
end

"""
    GameResidual(game)

BifurcationKit's `F(x, pin)` of `game`'s branch with the relative precision carried as the state's last
coordinate: `x = [u; ρ]`, `F = [sinh(u) - T(sinh(u), ρ P); ρ - pin]`, `P` the target precision. The logits
`z = sinh(u)` grow like `ρ` along the branch; `asinh` keeps their arclength commensurate with `ρ`'s near the end.
`pin` is BifurcationKit's continuation parameter, which only pins `ρ`.
"""
struct GameResidual{G <: LogitGame}
    game::G
end

function (residual::GameResidual)(x::AbstractVector{E}, pin::Real)::Vector{E} where {E <:
    Real}
    return branch_system(residual.game, x, pin).residual
end

"""
    GameSystem(game)

BifurcationKit's fused `FJ(x, pin)` of `game`'s branch: the residual and Jacobian from one `pass`
(`branch_system`). PALC's Newton method calls it in place of the residual and `J`.
"""
struct GameSystem{G <: LogitGame}
    game::G
end

function (system::GameSystem)(x::AbstractVector{E}, pin::Real)::Tuple{
    Vector{E},
    Matrix{E},
} where {E <: Real}
    filled = branch_system(system.game, x, pin)
    return filled.residual, filled.jacobian
end

"""BifurcationKit's `J(x, pin)` of `game`'s branch."""
struct GameJacobian{G <: LogitGame}
    game::G
end

function (jacobian::GameJacobian)(x::AbstractVector{E}, pin::Real)::Matrix{E} where {E <:
    Real}
    return branch_system(jacobian.game, x, pin).jacobian
end

"""The state coordinates `u` of a branch point `x = [u; ρ]`."""
state_of(x::AbstractVector)::AbstractVector = view(x, firstindex(x):(lastindex(x) - 1))

"""
    branch_system(game, x, pin)

`(; residual, jacobian)` of `game`'s branch at `x = [u; ρ]`: `∂F/∂u = (I - ∂T/∂z) diag(cosh u)` and
`∂F/∂ρ = -P ∂T/∂precision`, over the relative precision's own row `[0 … 0 1]`.
"""
function branch_system(game::LogitGame, x::AbstractVector{E}, pin::Real)::NamedTuple{
    (:residual, :jacobian),
    Tuple{Vector{E}, Matrix{E}},
} where {E <: Real}
    u = state_of(x)
    width = length(u)
    z = sinh.(u)
    relative_precision = x[end]
    pass = game.pass(z, precision_at(game, relative_precision))
    residual = Vector{E}(undef, width + 1)
    jacobian = zeros(E, width + 1, width + 1)
    for (slot, logit) in pairs(z)
        residual[slot] = logit - pass.response[slot]
        jacobian[slot, width + 1] = -pass.precision_slope[slot] * game.target_precision
        for (rival, scale) in pairs(cosh.(u))
            jacobian[slot, rival] = ((slot == rival) - pass.logit_slope[slot, rival]) *
                scale
        end
    end
    residual[end] = relative_precision - pin
    jacobian[end, end] = one(E)
    return (; residual, jacobian)
end

"""
    PinSlope(start)

`∂F/∂pin` of the branch residual `[F(u, ρ); ρ - pin]`, BifurcationKit's jet `R01`: `[0 … 0 -1]`, exactly, in
`start`'s element type and length. BifurcationKit copies the vector it gets, so one serves every call.
The constructor is inner because the field's default one would take `start` itself as the slope.
"""
struct PinSlope{E <: Real}
    slope::Vector{E}

    function PinSlope(start::AbstractVector{E})::PinSlope{E} where {E <: Real}
        return new{E}(setindex!(zeros(E, length(start)), -one(E), length(start)))
    end
end

function (pin_slope::PinSlope{E})(::AbstractVector{E}, ::Any)::Vector{E} where {E <: Real}
    return pin_slope.slope
end

"""
    branch_problem(game, start, pin; record_from_solution)

BifurcationKit's problem for `game`'s branch from `start = [u; ρ]` at `pin`, with the fused system and
exact `∂F/∂pin` supplied.
"""
function branch_problem(
    game::LogitGame,
    start::AbstractVector{E},
    pin::Real;
    record_from_solution::Any,
)::BifurcationKit.AbstractBifurcationProblem where {E <: Real}
    return BifurcationKit.BifurcationProblem(
        GameResidual(game),
        start,
        pin;
        J = GameJacobian(game),
        FJ = GameSystem(game),
        delta = one(E),
        inplace = false,
        R01 = PinSlope(start),
        record_from_solution,
    )
end

"""
    augmented_theta(θ, count)

`θ` for a branch of `count` state coordinates: PALC's arclength product `θ ⟨Δx, τx⟩ / length(x) + (1 - θ) Δρ τρ`
over the `count + 1` coordinates, equal to the original at `θ` over `count`. It must stay below 1: at 1
PALC weights the relative precision by zero, and its secant start divides by that zero.
"""
function augmented_theta(θ::T, count::Integer)::T where {T <: Real}
    augmented = θ * (count + 1) / count
    if !(augmented < 1)
        throw(ArgumentError(
            "a trace of width $count at θ = $θ has augmented θ = $augmented, which PALC needs below 1",
        ))
    end
    return augmented
end
