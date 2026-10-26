using LinearAlgebra: Diagonal

"""
    projection_step(game, coordinates, logits, relative_precision, state_weight)

One minimum-norm Gauss–Newton step from `x = [u; ρ]`, `u` the state coordinates of `logits`, onto `game`'s branch `F(x) = 0`, in PALC's norm
`θ ‖du‖² / k + (1 - θ) dρ²` (`θ` the `state_weight`, `k` the number of logits): `dx = -W⁻¹ Jᵀ (J W⁻¹ Jᵀ)⁻¹ F`.
It is the minimum-norm solution of `(J W^-½) y = -F` by pivoted QR, scaled back by `W^-½`; the normal matrix is never
formed. The step moves `ρ` as well, so it stays defined at a fold, where a correction at fixed `ρ` is singular.
"""
function projection_step(
    game::LogitGame,
    coordinates::LogitCoordinates,
    logits::AbstractVector{E},
    relative_precision::Real,
    state_weight::Real,
)::AbstractVector{E} where {E <: Real}
    if !(0 < state_weight < 1)
        throw(ArgumentError("state_weight must lie in (0, 1); got $state_weight"))
    end
    width = length(logits)
    point = [states_at(coordinates, logits); E(relative_precision)]
    system = branch_system!(BranchWorkspace(point), game, coordinates, point, relative_precision)
    jacobian = system.jacobian[1:width, :]
    scale = Diagonal([fill(E(sqrt(width / state_weight)), width); E(sqrt(1 / (1 - state_weight)))])
    return -(scale * ((jacobian * scale) \ system.residual[1:width]))
end

"""
    branch_projection(game, logits, relative_precision, settings)

`(; logits, relative_precision)`: the point one `projection_step` from `(logits, ρ)` toward `game`'s branch, in the
trace's coordinates and norm (`settings.coordinates`, `settings.state_weight`). The projection is first order: its residual is quadratic in the start's.
"""
function branch_projection(
    game::LogitGame,
    logits::AbstractVector{E},
    relative_precision::Real,
    settings::TraceSettings,
)::NamedTuple{(:logits, :relative_precision), Tuple{Vector{E}, E}} where {E <: Real}
    (; coordinates) = settings
    step = projection_step(game, coordinates, logits, relative_precision, settings.state_weight)
    moved = states_at(coordinates, logits) .+ state_of(step)
    return (; logits = logits_at(coordinates, moved), relative_precision = E(relative_precision) + last(step))
end
