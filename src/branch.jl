using BifurcationKit: BifurcationKit
using LinearAlgebra: mul!
"""
    LogitGame(pass!, target_precision)

A logit game whose equilibria solve `z = λ G(z, λ)` at precision `λ = 1 / temperature`, `G` the payoff gains the
logits scale. `pass!(gain, gain_slope, gain_precision_slope, z, λ)` writes `G(z, λ)`, its Jacobian `∂G/∂z` and its
slope `∂G/∂λ` into the three arrays, from one evaluation, and returns `nothing`. The arrays are views into the
trace's own residual and Jacobian (`BranchWorkspace`), so a pass allocates nothing of its own. The flat
equilibrium `z = 0` is the only root at zero precision. `target_precision` is the precision a trace ends at.
"""
struct LogitGame{P, T <: Real}
    pass!::P
    target_precision::T
end

"""The precision at `relative_precision`: the fraction of `game.target_precision` the trace has reached."""
function precision_at(game::LogitGame, relative_precision::R)::typeof(
    relative_precision * game.target_precision,
) where {R <: Real}
    return relative_precision * game.target_precision
end

"""
    BranchWorkspace(point)

A residual, Jacobian and logits for branch points like `point = [u; ρ]`, refilled by every call of the one
callable that owns it. BifurcationKit consumes each result before calling that callable again, so the fused system
and the Jacobian each own one workspace per trace; a shared one would let `J` overwrite a residual `FJ` returned.
"""
struct BranchWorkspace{E <: Real}
    residual::Vector{E}
    jacobian::Matrix{E}
    logits::Vector{E}
end

function BranchWorkspace(point::AbstractVector{E})::BranchWorkspace{E} where {E <: Real}
    return BranchWorkspace(
        Vector{E}(undef, length(point)),
        Matrix{E}(undef, length(point), length(point)),
        Vector{E}(undef, length(point) - 1),
    )
end

"""
    GameResidual(game, coordinates)

BifurcationKit's `F(x, pin)` of `game`'s branch with the relative precision carried as the state's last
coordinate: `x = [u; ρ]` with `u` in `coordinates`, `F = [z - λ G(z, λ); ρ - pin]` at the logits `z` of `u` and
`λ = ρ P`, `P` the target precision; for `PayoffGaps`, `F = [u - P G(ρ u, λ); ρ - pin]`, which stays regular at
`ρ = 0`. `pin` is BifurcationKit's continuation parameter, which only pins `ρ`. Each call fills a fresh
`BranchWorkspace`: BifurcationKit keeps its problem's `F` apart from the fused system.
"""
struct GameResidual{G <: LogitGame}
    game::G
    coordinates::TraceCoordinates
end

function (residual::GameResidual)(point::AbstractVector{E}, pin::Real)::AbstractVector{E} where {E <: Real}
    return branch_system!(BranchWorkspace(point), residual.game, residual.coordinates, point, pin).residual
end

"""
    GameSystem(game, coordinates, workspace)

BifurcationKit's fused `FJ(x, pin)` of `game`'s branch: the residual and Jacobian from one `pass!`, written into
`workspace` (`branch_system!`). PALC's Newton method calls it in place of the residual and `J`.
"""
struct GameSystem{G <: LogitGame, W <: BranchWorkspace}
    game::G
    coordinates::TraceCoordinates
    workspace::W
end

function (system::GameSystem)(point::AbstractVector{E}, pin::Real)::Tuple{
    Vector{E},
    Matrix{E},
} where {E <: Real}
    filled = branch_system!(system.workspace, system.game, system.coordinates, point, pin)
    return filled.residual, filled.jacobian
end

"""BifurcationKit's `J(x, pin)` of `game`'s branch in `coordinates`, written into `workspace`."""
struct GameJacobian{G <: LogitGame, W <: BranchWorkspace}
    game::G
    coordinates::TraceCoordinates
    workspace::W
end

function (jacobian::GameJacobian)(point::AbstractVector{E}, pin::Real)::AbstractMatrix{E} where {E <: Real}
    return branch_system!(jacobian.workspace, jacobian.game, jacobian.coordinates, point, pin).jacobian
end

"""The state coordinates `u` of a branch point `x = [u; ρ]`."""
state_of(point::AbstractVector)::AbstractVector = view(
    point,
    firstindex(point):(lastindex(point) - 1),
)

"""
    branch_system!(workspace, game, coordinates, x, pin)

`workspace` with `game`'s branch residual and Jacobian at `x = [u; ρ]` in `coordinates`, over the relative
precision's own row `[0 … 0 1]`. With `λ = ρ P` and `dz/du` the logits' slope (`cosh u`, or 1):
`∂F/∂u = (I - λ ∂G/∂z) diag(dz/du)` and `∂F/∂ρ = -P (G + λ ∂G/∂λ)`; for `PayoffGaps`, `∂F/∂u = I - λ ∂G/∂z` and
`∂F/∂ρ = -P (∂G/∂z u + P ∂G/∂λ)`. The pass writes `G` and its slopes into the blocks that hold `F` and its Jacobian.
"""
function branch_system!(
    workspace::BranchWorkspace{E},
    game::LogitGame,
    coordinates::TraceCoordinates,
    point::AbstractVector{E},
    pin::Real,
)::typeof(workspace) where {E <: Real}
    (; residual, jacobian, logits) = workspace
    state = state_of(point)
    width = length(state)
    relative_precision = point[end]
    trace_logits!(logits, coordinates, state, relative_precision)
    precision = precision_at(game, relative_precision)
    target = game.target_precision
    gain = view(residual, 1:width)
    # fatou-ignore call-arity: Base `view` takes any number of indices (jolars/fatou#125)
    gain_slope = view(jacobian, 1:width, 1:width)
    # fatou-ignore call-arity: Base `view` takes any number of indices (jolars/fatou#125)
    precision_column = view(jacobian, 1:width, width + 1)
    game.pass!(gain, gain_slope, precision_column, logits, precision)
    # `∂F/∂ρ` first: it reads `G` and `∂G/∂z`, which the next lines overwrite.
    if coordinates == PayoffGaps
        mul!(precision_column, gain_slope, state, -target, -target^2)
    else
        precision_column .= .-target .* (gain .+ precision .* precision_column)
    end
    for (column, coordinate) in pairs(state)
        scale = logit_scale(coordinates, coordinate)
        for row in axes(gain_slope, 1)
            gain_slope[row, column] = ((row == column) - precision * gain_slope[row, column]) * scale
        end
    end
    if coordinates == PayoffGaps
        gain .= state .- target .* gain
    else
        gain .= logits .- precision .* gain
    end
    # fatou-ignore call-arity: Base `view` takes any number of indices (jolars/fatou#125)
    fill!(view(jacobian, width + 1, 1:width), zero(E))
    jacobian[end, end] = one(E)
    residual[end] = relative_precision - pin
    return workspace
end

"""
    trace_start(game, coordinates, width, element)

The branch's start `[u; 0]` at zero precision: the flat logits, or for `PayoffGaps` the gains `P G(0, 0)`, which
is minus the gap residual at `u = 0`.
"""
function trace_start(
    game::LogitGame,
    coordinates::TraceCoordinates,
    width::Integer,
    ::Type{E},
)::Vector{E} where {E <: Real}
    start = zeros(E, width + 1)
    if coordinates == PayoffGaps
        state_of(start) .= .-state_of(branch_system!(BranchWorkspace(start), game, coordinates, start, zero(E)).residual)
    end
    return start
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

function (pin_slope::PinSlope{E})(::AbstractVector{E}, ::Any)::AbstractVector{E} where {E <:
    Real}
    return pin_slope.slope
end

"""
    branch_problem(game, coordinates, start, pin; record_from_solution)

BifurcationKit's problem for `game`'s branch in `coordinates` from `start = [u; ρ]` at `pin`, with the fused system
and exact `∂F/∂pin` supplied. The system and Jacobian each own a `BranchWorkspace`.
"""
function branch_problem(
    game::LogitGame,
    coordinates::TraceCoordinates,
    start::AbstractVector{E},
    pin::Real;
    record_from_solution::Any,
)::BifurcationKit.AbstractBifurcationProblem where {E <: Real}
    return BifurcationKit.BifurcationProblem(
        GameResidual(game, coordinates),
        start,
        pin;
        J = GameJacobian(game, coordinates, BranchWorkspace(start)),
        FJ = GameSystem(game, coordinates, BranchWorkspace(start)),
        delta = one(E),
        inplace = false,
        R01 = PinSlope(start),
        record_from_solution,
    )
end

"""
    augmented_theta(state_weight, count)

The state weight for a branch of `count` state coordinates: PALC's arclength product `θ ⟨Δx, τx⟩ / length(x) + (1 - θ) Δρ τρ`
over the `count + 1` coordinates, equal to the original at `state_weight` over `count`. It must stay below 1: at 1
PALC weights the relative precision by zero, and its secant start divides by that zero.
"""
function augmented_theta(state_weight::T, count::Integer)::T where {T <: Real}
    augmented = state_weight * (count + 1) / count
    if !(augmented < 1)
        throw(ArgumentError(
            "a trace of width $count at state weight $state_weight has augmented θ = $augmented, which PALC needs below 1",
        ))
    end
    return augmented
end
