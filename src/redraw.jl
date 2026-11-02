"""A `redraw` hook that never swaps the game."""
struct NoRedraw end

(::NoRedraw)(::AbstractVector, ::Real)::Nothing = nothing

"""
    RedrawSlot(game, pending, step_length)

The game a trace runs on, whether `redraw` swapped it since the last restart (`pending`), and the arclength step the
trace last held, which a restart carries on.
"""
mutable struct RedrawSlot{G <: LogitGame, E <: Real}
    # Mutable because BifurcationKit's finalise callback returns only a `Bool`.
    game::G
    pending::Bool
    step_length::E
end

"""
    swapped!(slot, redrawn)

Whether `redrawn` swaps `slot`'s game: `nothing` and the current game do not. A swap sets `pending`. A game of
another type, or with another target precision, is a caller error: the trace's relative precision would change meaning.
"""
swapped!(::RedrawSlot, ::Nothing)::Bool = false

function swapped!(slot::RedrawSlot{G}, redrawn::G)::Bool where {G <: LogitGame}
    if redrawn === slot.game
        return false
    end
    if redrawn.target_precision != slot.game.target_precision
        throw(ArgumentError(
            "a redrawn game has target precision $(redrawn.target_precision), not the trace's $(slot.game.target_precision)",
        ))
    end
    slot.game = redrawn
    slot.pending = true
    return true
end

"""
    restart_tangent(game, point, old_tangent, dot, state_weight, orientation)

The tangent of `game`'s branch at `point = [w; ρ]` that keeps the trace's direction: the solution of the bordered
system `[∂F/∂x; bᵀ] τ = [0; 1]`, whose border `b` is `old_tangent` weighted as PALC's product under the trace's `dot`
at PALC state weight `state_weight` weights it (`palc_border`), with the sign that makes its `orientation` product
(`PlainOrientation`) with the old tangent positive.
"""
function restart_tangent(
    game::LogitGame,
    point::AbstractVector{E},
    old_tangent::AbstractVector{E},
    dot::BifurcationKit.DotTheta,
    state_weight::E,
    orientation::PlainOrientation{E},
)::AbstractVector{E} where {E <: Real}
    width = length(point) - 1
    jacobian = branch_system!(
        BranchWorkspace(game, point),
        game,
        point,
        last(point),
    ).jacobian
    border = palc_border(dot, state_weight, old_tangent)
    bordered = [@view(jacobian[1:width, :]); transpose(border)]
    tangent = bordered \ setindex!(zeros(E, width + 1), one(E), width + 1)
    return sign(orientation(
        old_tangent,
        tangent,
        last(old_tangent),
        last(tangent),
        state_weight,
    )) .*
        tangent
end
