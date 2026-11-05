using LinearAlgebra: LU, lu
"""A `redraw` hook that never swaps the game."""
struct NoRedraw end

(::NoRedraw)(::AbstractVector, ::Real, ::Any)::Nothing = nothing

"""
    ShiftedGame(game, point, factorization, shift, shift_move, within)

A candidate `game` at the accepted point `point = [w; ρ]` (`RootShift`): its bordered Jacobian `[∂F/∂x; bᵀ]` there,
factored, its root `shift` `δ`, that shift's largest logit move `shift_move`, and whether the move is `within` the
check's limit. A non-finite map or Jacobian at `point` has no factorization (`nothing`), a NaN shift and is never
`within`; a restart on it fails as LAPACK's check of a non-finite matrix does, at the swap point.
"""
struct ShiftedGame{G <: LogitGame, E <: Real}
    game::G
    point::Vector{E}
    factorization::Union{Nothing, LU{E, Matrix{E}, Vector{Int}}}
    shift::Vector{E}
    shift_move::E
    within::Bool
end

"""
    RedrawSlot(game, pending, step_length)

The game a trace runs on, whether `redraw` swapped it since the last restart (`pending`), the arclength step the
trace last held, which a restart carries on, and the swapped game's factored bordered Jacobian at the swap point
(`ShiftedGame`), from which the restart takes its tangent.
"""
mutable struct RedrawSlot{G <: LogitGame, E <: Real}
    # Mutable because BifurcationKit's finalise callback returns only a `Bool`.
    game::G
    pending::Bool
    step_length::E
    factorization::Union{Nothing, LU{E, Matrix{E}, Vector{Int}}}
end

function RedrawSlot(game::G, pending::Bool, step_length::E)::RedrawSlot{G, E} where {G <: LogitGame, E <: Real}
    return RedrawSlot{G, E}(game, pending, step_length, nothing)
end

"""
    restart_tangent(factorization, old_tangent, state_weight, orientation)

The tangent of the swapped game's branch at the swap point that keeps the trace's direction: the solution of the
bordered system `[∂F/∂x; bᵀ] τ = [0; 1]`, held factored (`ShiftedGame`), with the sign that makes its `orientation`
product (`PlainOrientation`) with `old_tangent` positive at PALC state weight `state_weight`.
"""
function restart_tangent(
        factorization::LU{E},
        old_tangent::AbstractVector{E},
        state_weight::E,
        orientation::PlainOrientation{E},
    )::AbstractVector{E} where {E <: Real}
    tangent = factorization \ setindex!(zero(old_tangent), one(E), lastindex(old_tangent))
    return sign(
        orientation(
            old_tangent,
            tangent,
            last(old_tangent),
            last(tangent),
            state_weight,
        )
    ) .*
        tangent
end

"""
    RootShift(point, border, limit)
    RootShift(point, step, border, settings)

The check a `redraw` hook receives at the accepted point `point = [w; ρ]`. Called on a candidate game, it returns
the `ShiftedGame`: one evaluation of the game's map and Jacobian at `point`, the bordered system under the
corrector's `border` factored once, and the root shift `δ` from `[∂F/∂x; borderᵀ] δ = [-F(point); 0]`, one Newton
step of the bordered corrector. The old map's residual at an accepted point is within the corrector's tolerance, so
`-F(point)` stands for `F_old(point) - F_new(point)`. The shift is `within` when it moves no logit further than
`limit`: `settings.shift_share` of the corrector's reach on the next step, the predictor `step` (`CorrectorSettings`),
or `nothing` when the reach is off, and every game passes. A hook returns the `ShiftedGame` to install its game, and
the restart reuses its factorization.
"""
struct RootShift{E <: Real, L <: Union{Nothing, E}}
    point::Vector{E}
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
    return RootShift(collect(point), collect(border), scaled_limit(scaled_limit(settings.reach, move), settings.shift_share))
end

function (check::RootShift{E})(game::LogitGame)::ShiftedGame{typeof(game), E} where {E <: Real}
    (; point, border) = check
    width = length(point) - 1
    system = branch_system!(BranchWorkspace(game, point), game, point, last(point))
    bordered = [@view(system.jacobian[1:width, :]); transpose(border)]
    if !(all(isfinite, bordered) && all(isfinite, system.residual))
        return ShiftedGame(game, point, nothing, fill(E(NaN32), width + 1), E(NaN32), false)
    end
    factorization = lu(bordered)
    shift = factorization \ [.-system.residual[1:width]; zero(E)]
    move = logit_change(point, point .+ shift)
    return ShiftedGame(game, point, factorization, shift, move, within_limit(check.limit, move))
end

"""
    swapped!(slot, check, redrawn)

Whether a `redraw` hook's result `redrawn` swaps `slot`'s game: `nothing` and the current game do not; a new game
swaps as the `ShiftedGame` that `check` (`RootShift`) builds for it, and a `ShiftedGame` the hook returned must come
from `check`. A swap sets `pending` and the factorization the restart reads. A game of another type, or with another
target precision, is a caller error: the trace's relative precision would change meaning.
"""
swapped!(::RedrawSlot, ::RootShift, ::Nothing)::Bool = false

function swapped!(slot::RedrawSlot, check::RootShift, redrawn::LogitGame)::Bool
    if redrawn === slot.game
        return false
    end
    return swapped!(slot, check, check(redrawn))
end

function swapped!(slot::RedrawSlot{G, E}, check::RootShift, redrawn::ShiftedGame{G})::Bool where {G <: LogitGame, E <: Real}
    if redrawn.point !== check.point
        throw(ArgumentError("a redraw hook returned a ShiftedGame built at another point than this one"))
    end
    if redrawn.game === slot.game
        return false
    end
    if redrawn.game.target_precision != slot.game.target_precision
        throw(
            ArgumentError(
                "a redrawn game has target precision $(redrawn.game.target_precision), not the trace's $(slot.game.target_precision)",
            )
        )
    end
    slot.game = redrawn.game
    slot.pending = true
    slot.factorization = redrawn.factorization
    return true
end
