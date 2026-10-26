using LinearAlgebra: Diagonal

"""
    projection_step(game, logits, relative_precision, state_weight)

One minimum-norm Gauss–Newton step from `x = [logits / ρ; ρ]` onto `game`'s branch `F(x) = 0` in payoff gaps, in
PALC's norm `θ ‖dw‖² / k + (1 - θ) dρ²` (`θ` the `state_weight`, `k` the number of logits): `dx = -W⁻¹ Jᵀ (J W⁻¹ Jᵀ)⁻¹ F`.
It is the minimum-norm solution of `(J W^-½) y = -F` by pivoted QR, scaled back by `W^-½`; the normal matrix is never
formed. The step moves `ρ` as well, so it stays defined at a fold, where a correction at fixed `ρ` is singular.
Payoff gaps have no point at `ρ = 0`, so `ρ` must be positive.
"""
function projection_step(
    game::LogitGame,
    logits::AbstractVector{E},
    relative_precision::Real,
    state_weight::Real,
)::AbstractVector{E} where {E <: Real}
    if !(0 < state_weight < 1)
        throw(ArgumentError("state_weight must lie in (0, 1); got $state_weight"))
    end
    if !(relative_precision > 0)
        throw(ArgumentError("payoff gaps need a positive relative precision; got $relative_precision"))
    end
    width = length(logits)
    point = [logits ./ E(relative_precision); E(relative_precision)]
    system = branch_system!(BranchWorkspace(point), game, point, relative_precision)
    jacobian = system.jacobian[1:width, :]
    scale = Diagonal([fill(E(sqrt(width / state_weight)), width); E(sqrt(1 / (1 - state_weight)))])
    return -(scale * ((jacobian * scale) \ system.residual[1:width]))
end

"""
    branch_projection(game, logits, relative_precision, settings)

`(; logits, relative_precision)`: the point one `projection_step` from `(logits, ρ)` toward `game`'s branch, in the
trace's norm (`settings.state_weight`). The projection is first order: its residual is quadratic in the start's.
"""
function branch_projection(
    game::LogitGame,
    logits::AbstractVector{E},
    relative_precision::Real,
    settings::TraceSettings,
)::NamedTuple{(:logits, :relative_precision), Tuple{Vector{E}, E}} where {E <: Real}
    step = projection_step(game, logits, relative_precision, settings.state_weight)
    moved = [logits ./ E(relative_precision); E(relative_precision)] .+ step
    return (; logits = logits_at(moved), relative_precision = last(moved))
end
