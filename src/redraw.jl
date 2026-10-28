"""A `redraw` hook that never swaps the game."""
struct NoRedraw end

(::NoRedraw)(::AbstractVector, ::Real)::Nothing = nothing

"""
    RedrawSlot(game, before, pending, step_length)

The game a trace runs on, the game it ran on before the last swap (`before`), whether `redraw` swapped it since the
last restart (`pending`), and the arclength step the trace last held, which a restart carries on.
"""
mutable struct RedrawSlot{G <: LogitGame, E <: Real}
    # Mutable because BifurcationKit's finalise callback returns only a `Bool`, as for `TraceProgress`.
    game::G
    before::G
    pending::Bool
    step_length::E
    # EXPERIMENT: Gambit's step h in its coordinates (path.cc), NaN when the trace runs another step control.
    gambit_step::E
end

function RedrawSlot(game::G, before::G, pending::Bool, step_length::E)::RedrawSlot{G, E} where {G <: LogitGame, E <: Real}
    return RedrawSlot{G, E}(game, before, pending, step_length, E(NaN))
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
    slot.before = slot.game
    slot.game = redrawn
    slot.pending = true
    return true
end

"""
    redraw_crossing(before, after, point, settings)

The `CrossingResult` of `homotopy_crossing` from `point = [w; ρ]`, a root of game `before`, to a root of game
`after` at the same `ρ`, under the trace's PALC (`palc_algorithm`), step controls and corrector (`CorrectorReach`). The
crossing replaces the map at fixed `ρ`, so it lands on the root its zero continues to, or fails. A start that would
not correct (`uncorrected_start`) or a broken linear solve (`solver_blowup`) ends it as reason `:solver_blowup`.
"""
function redraw_crossing(
    before::LogitGame,
    after::LogitGame,
    point::AbstractVector{E},
    settings::TraceSettings{E},
)::CrossingResult{E} where {E <: Real}
    relative_precision = last(point)
    corrector = CorrectorReach(AcceptAll(), settings.corrector, settings.geometry, similar(point), similar(point), zero(E), GambitDistance(before.pass!, E(before.target_precision)))
    try
        return homotopy_crossing(
            branch_problem(before, point, relative_precision; record_from_solution = trace_record_precision),
            branch_problem(after, point, relative_precision; record_from_solution = trace_record_precision),
            point,
            # The orientation check flips the tangent at a branch point, which would read a crossing as a fold.
            palc_algorithm(settings, length(point) - 1, before.target_precision; orientation_check = false, step_distance = gambit_step_distance(settings, before)),
            continuation_options(settings, settings.first_step, settings.max_steps, settings.first_tangent_divisor);
            callback_newton = corrector,
        )
    catch exception
        if !uncorrected_start(exception) && !solver_blowup(exception)
            rethrow()
        end
        return CrossingResult(collect(point), :solver_blowup, 0)
    end
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
    jacobian = branch_system!(BranchWorkspace(game, point), game, point, last(point)).jacobian
    border = palc_border(dot, state_weight, old_tangent)
    # fatou-ignore call-arity: Base `view` takes any number of indices (jolars/fatou#125)
    bordered = [view(jacobian, 1:width, :); transpose(border)]
    tangent = bordered \ setindex!(zeros(E, width + 1), one(E), width + 1)
    return sign(orientation(old_tangent, tangent, last(old_tangent), last(tangent), state_weight)) .* tangent
end
