using LinearAlgebra: Diagonal

"""
    projection_step(game, geometry, logits, relative_precision)

One minimum-norm Gauss–Newton step from `x = [u; ρ]`, `u` the `logits` in `geometry.coordinates`, onto `game`'s branch
`F(x) = 0`, in `geometry`'s plain norm `W` (`coordinate_weight`, `level_weight`): `dx = -W⁻¹ Jᵀ (J W⁻¹ Jᵀ)⁻¹ F`.
It is the minimum-norm solution of `(J W^-½) y = -F` by pivoted QR, scaled back by `W^-½`; the normal matrix is never
formed. The step moves `ρ` as well, so it stays defined at a fold, where a correction at fixed `ρ` is singular.
"""
function projection_step(
    game::LogitGame,
    geometry::TraceGeometry,
    logits::AbstractVector{E},
    relative_precision::Real,
)::AbstractVector{E} where {E <: Real}
    if !(0 < geometry.state_weight < 1)
        throw(ArgumentError("state_weight must lie in (0, 1); got $(geometry.state_weight)"))
    end
    width = length(logits)
    point = [trace_state(geometry.coordinates, logits, relative_precision); E(relative_precision)]
    system = branch_system!(BranchWorkspace(point), game, geometry.coordinates, point, relative_precision)
    jacobian = system.jacobian[1:width, :]
    weights = [fill(E(coordinate_weight(geometry, width)), width); E(level_weight(geometry))]
    scale = Diagonal(inv.(sqrt.(weights)))
    return -(scale * ((jacobian * scale) \ system.residual[1:width]))
end

"""
    branch_projection(game, logits, relative_precision, settings)

`(; logits, relative_precision)`: the point one `projection_step` from `(logits, ρ)` toward `game`'s branch, in the
trace's geometry (`settings.geometry`). The projection is first order: its residual is quadratic in the start's.
"""
function branch_projection(
    game::LogitGame,
    logits::AbstractVector{E},
    relative_precision::Real,
    settings::TraceSettings,
)::NamedTuple{(:logits, :relative_precision), Tuple{Vector{E}, E}} where {E <: Real}
    (; coordinates) = settings.geometry
    step = projection_step(game, settings.geometry, logits, relative_precision)
    projected = E(relative_precision) + last(step)
    state = trace_state(coordinates, logits, relative_precision) .+ state_of(step)
    return (; logits = trace_logits(coordinates, state, projected), relative_precision = projected)
end
