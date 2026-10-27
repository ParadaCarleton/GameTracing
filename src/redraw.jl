using SciMLBase: SciMLBase, NonlinearFunction, NonlinearProblem

"""A `redraw` hook that never swaps the game."""
struct NoRedraw end

(::NoRedraw)(::AbstractVector, ::Real)::Nothing = nothing

"""
    RedrawSlot(game, pending, step_length)

The game a trace runs on, whether `redraw` swapped it since the last restart (`pending`), and the arclength step
the trace last held, which a restart carries on.
"""
mutable struct RedrawSlot{G <: LogitGame, E <: Real}
    # Mutable because BifurcationKit's finalise callback returns only a `Bool`, as for `TraceProgress`.
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
    LevelSystem(game, relative_precision, workspace)

`game`'s branch residual `w - P G(ρ w, ρ P)` in the payoff gaps `w` at the fixed relative precision `ρ`, and its
Jacobian `I - λ ∂G/∂z`, as NonlinearSolve's in-place functions (`level_residual!`, `level_jacobian!`).
"""
struct LevelSystem{G <: LogitGame, E <: Real, W <: BranchWorkspace{E}}
    game::G
    relative_precision::E
    workspace::W
end

"""`system.workspace` filled at the gaps `gaps` and `system`'s relative precision."""
function level_system!(system::LevelSystem, gaps::AbstractVector{E})::typeof(system.workspace) where {E <: Real}
    point = [gaps; system.relative_precision]
    return branch_system!(system.workspace, system.game, point, system.relative_precision)
end

function level_residual!(residual::AbstractVector{E}, gaps::AbstractVector{E}, system::LevelSystem)::Nothing where {E <: Real}
    residual .= payoff_gaps(level_system!(system, gaps).residual)
    return nothing
end

function level_jacobian!(jacobian::AbstractMatrix{E}, gaps::AbstractVector{E}, system::LevelSystem)::Nothing where {E <: Real}
    jacobian .= view(level_system!(system, gaps).jacobian, axes(jacobian)...)
    return nothing
end

"""
    level_correction(game, point, settings, solver)

`(; point, corrected, residual, distance, retcode)`: the root of `game` at the fixed relative precision of
`point = [w; ρ]` that NonlinearSolve's `solver` reaches from `point`. It is `corrected` when its sup-norm residual is
under `corrector_tolerance` and its logits lie within `settings.corrector.redraw_reach` (Euclidean) of `point`'s;
a farther root lies on another branch.
"""
function level_correction(
    game::LogitGame,
    point::AbstractVector{E},
    settings::TraceSettings{E},
    solver::Any,
)::NamedTuple where {E <: Real}
    relative_precision = last(point)
    solution = SciMLBase.solve(
        NonlinearProblem(
            NonlinearFunction{true}(level_residual!; jac = level_jacobian!),
            collect(payoff_gaps(point)),
            LevelSystem(game, relative_precision, BranchWorkspace(game, point)),
        ),
        solver;
        abstol = settings.corrector.residual_tolerance,
        maxiters = settings.corrector.newton_iterations,
    )
    corrected_point = [solution.u; relative_precision]
    workspace = branch_system!(BranchWorkspace(game, point), game, corrected_point, relative_precision)
    residual = maximum(abs, payoff_gaps(workspace.residual))
    distance = relative_precision * sqrt(sum(abs2, solution.u .- payoff_gaps(point)))
    tolerance = corrector_tolerance(
        settings.corrector.residual_tolerance,
        settings.corrector.tolerance_floor_ulps,
        corrected_point,
    )
    return (;
        point = corrected_point,
        corrected = residual < tolerance && distance <= settings.corrector.redraw_reach,
        residual,
        distance,
        solution.retcode,
    )
end

"""
    restart_tangent(game, point, old_tangent, dot, state_weight)

The tangent of `game`'s branch at `point = [w; ρ]` that keeps the trace's direction: the solution of the bordered
system `[∂F/∂x; bᵀ] τ = [0; 1]`, whose border `b` is `old_tangent` weighted as PALC's product under the trace's `dot`
at PALC state weight `state_weight` weights it (`palc_border`). So `τ` has PALC product one with the old tangent.
"""
function restart_tangent(
    game::LogitGame,
    point::AbstractVector{E},
    old_tangent::AbstractVector{E},
    dot::BifurcationKit.DotTheta,
    state_weight::E,
)::AbstractVector{E} where {E <: Real}
    width = length(point) - 1
    jacobian = branch_system!(BranchWorkspace(game, point), game, point, last(point)).jacobian
    border = palc_border(dot, state_weight, old_tangent)
    # fatou-ignore call-arity: Base `view` takes any number of indices (jolars/fatou#125)
    bordered = [view(jacobian, 1:width, :); transpose(border)]
    return bordered \ setindex!(zeros(E, width + 1), one(E), width + 1)
end
