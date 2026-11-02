"""A `redraw` hook that never swaps the game."""
struct NoRedraw end

(::NoRedraw)(::AbstractVector, ::Real, ::Any)::Nothing = nothing

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

"""
    root_shift(game, point, border)

The first-order move of the root at `point = [w; ρ]` when the map becomes `game`: the solution `δ` of
`[∂F/∂x; borderᵀ] δ = [-F(point); 0]` over `game`'s rows, one Newton step of the bordered corrector from `point`.
The old map's residual at an accepted point is within the corrector's tolerance, so `-F(point)` stands for
`F_old(point) - F_new(point)` at one map evaluation.
"""
function root_shift(
    game::LogitGame,
    point::AbstractVector{E},
    border::AbstractVector{E},
)::AbstractVector{E} where {E <: Real}
    width = length(point) - 1
    system = branch_system!(BranchWorkspace(game, point), game, point, last(point))
    bordered = [@view(system.jacobian[1:width, :]); transpose(border)]
    return bordered \ [.-system.residual[1:width]; zero(E)]
end

"""
    RootShift(point, step, border, limit)

The check a `redraw` hook receives at the accepted point `point = [w; ρ]`: called on a candidate game, whether that
game's `root_shift` from `point` (under the corrector's `border`) moves no logit further than `limit`, a share of the
corrector's reach on the next step, the predictor `step` (`CorrectorSettings`). `limit` is `nothing` when the reach
is off, and every game passes.
"""
struct RootShift{E <: Real, L <: Union{Nothing, E}}
    point::Vector{E}
    step::Vector{E}
    border::Vector{E}
    limit::L
end

function RootShift(
    point::AbstractVector{E},
    step::AbstractVector{E},
    border::AbstractVector{E},
    settings::CorrectorSettings{E},
)::RootShift{E} where {E <: Real}
    move = logit_change(point, point .+ step)
    return RootShift(
        collect(point),
        collect(step),
        collect(border),
        scaled_limit(scaled_limit(settings.reach, move), settings.shift_share),
    )
end

"""The largest logit move of `game`'s `root_shift` from `check`'s point."""
function shift_move(check::RootShift{E}, game::LogitGame)::E where {E <: Real}
    return logit_change(check.point, check.point .+ root_shift(game, check.point, check.border))
end

(check::RootShift)(game::LogitGame)::Bool = within_limit(check.limit, shift_move(check, game))
