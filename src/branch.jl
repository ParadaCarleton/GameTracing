"""
    LogitGame(pass, temperature)

A logit game whose equilibria solve `z = T(z, τ)`: `pass(z, τ)` returns
`(; image, logit_slope, precision_slope)`, the map's value `T(z, τ)`, its Jacobian `∂T/∂z` and its slope
`∂T/∂λ` in the precision `λ = 1 / τ`, from one evaluation. `temperature` is the temperature `τ` a trace ends at.
`pass` must accept `τ = Inf` (precision zero), where `T` is zero.
"""
struct LogitGame{P, T <: Real}
    pass::P
    temperature::T
end

"""The temperature at a trace's `level`: `temperature / level`, the end's at level one and infinite at the start."""
level_temperature(temperature::Real, level::Real)::promote_type(
    typeof(temperature),
    typeof(level),
) = temperature / level

"""
    GameResidual(game)

BifurcationKit's `F(x, λ)` of `game`'s branch with its level carried as the state's last coordinate:
`x = [u; t]`, `F = [sinh(u) - T(sinh(u), temperature / t); t - λ]`. The logits `z = sinh(u)` grow like `t`
along the branch; `asinh` keeps their arclength commensurate with `t`'s near the end.
"""
struct GameResidual{G <: LogitGame}
    game::G
end

function (residual::GameResidual)(x::AbstractVector{E}, λ::Real)::Vector{E} where {E <:
    Real}
    return branch_system(residual.game, x, λ).residual
end

"""
    GameSystem(game)

BifurcationKit's fused `FJ(x, λ)` of `game`'s branch: the residual and Jacobian from one `pass`
(`branch_system`). PALC's Newton method calls it in place of the residual and `J`.
"""
struct GameSystem{G <: LogitGame}
    game::G
end

function (system::GameSystem)(x::AbstractVector{E}, λ::Real)::Tuple{
    Vector{E},
    Matrix{E},
} where {E <: Real}
    filled = branch_system(system.game, x, λ)
    return filled.residual, filled.jacobian
end

"""BifurcationKit's `J(x, λ)` of `game`'s branch."""
struct GameJacobian{G <: LogitGame}
    game::G
end

function (jacobian::GameJacobian)(x::AbstractVector{E}, λ::Real)::Matrix{E} where {E <:
    Real}
    return branch_system(jacobian.game, x, λ).jacobian
end

"""The state coordinates `u` of a branch point `x = [u; t]`."""
state_of(x::AbstractVector)::AbstractVector = view(x, firstindex(x):(lastindex(x) - 1))

"""
    branch_system(game, x, λ)

`(; residual, jacobian)` of `game`'s branch at `x = [u; t]`: `∂F/∂u = (I - ∂T/∂z) diag(cosh u)`,
`∂F/∂t = -∂T/∂λ / temperature` (the precision is `λ = t / temperature`), over the level's own row `[0 … 0 1]`.
"""
function branch_system(game::LogitGame, x::AbstractVector{E}, λ::Real)::NamedTuple{
    (:residual, :jacobian),
    Tuple{Vector{E}, Matrix{E}},
} where {E <: Real}
    u = state_of(x)
    width = length(u)
    z = sinh.(u)
    level = x[end]
    pass = game.pass(z, level_temperature(game.temperature, level))
    residual = Vector{E}(undef, width + 1)
    jacobian = zeros(E, width + 1, width + 1)
    for (slot, logit) in pairs(z)
        residual[slot] = logit - pass.image[slot]
        jacobian[slot, width + 1] = -pass.precision_slope[slot] / game.temperature
        for (rival, scale) in pairs(cosh.(u))
            jacobian[slot, rival] = ((slot == rival) - pass.logit_slope[slot, rival]) *
                scale
        end
    end
    residual[end] = level - λ
    jacobian[end, end] = one(E)
    return (; residual, jacobian)
end

"""
    LevelSlope(start)

`∂F/∂λ` of the branch residual `[F(u, t); t - λ]`, BifurcationKit's jet `R01`: `[0 … 0 -1]`, exactly, in
`start`'s element type and length. BifurcationKit copies the vector it gets, so one serves every call.
The constructor is inner because the field's default one would take `start` itself as the slope.
"""
struct LevelSlope{E <: Real}
    slope::Vector{E}

    function LevelSlope(start::AbstractVector{E})::LevelSlope{E} where {E <: Real}
        return new{E}(setindex!(zeros(E, length(start)), -one(E), length(start)))
    end
end

function (level_slope::LevelSlope{E})(::AbstractVector{E}, ::Any)::Vector{E} where {E <:
    Real}
    return level_slope.slope
end

"""
    branch_problem(game, start, level; record_from_solution)

BifurcationKit's problem for `game`'s branch from `start = [u; t]` at `level`, with the fused system and
exact `∂F/∂λ` supplied.
"""
function branch_problem(
    game::LogitGame,
    start::AbstractVector{E},
    level::Real;
    record_from_solution::Any,
)::BifurcationKit.AbstractBifurcationProblem where {E <: Real}
    return BifurcationKit.BifurcationProblem(
        GameResidual(game),
        start,
        level;
        J = GameJacobian(game),
        FJ = GameSystem(game),
        delta = one(E),
        inplace = false,
        R01 = LevelSlope(start),
        record_from_solution,
    )
end

"""
    augmented_theta(θ, count)

`θ` for a branch of `count` state coordinates: PALC's arclength product `θ ⟨Δx, τx⟩ / length(x) + (1 - θ) Δλ τλ`
over the `count + 1` coordinates, equal to the original at `θ` over `count`. It must stay below 1: at 1
PALC weights the level by zero, and its secant start divides by that zero.
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
