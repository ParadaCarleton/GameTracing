using BifurcationKit: BifurcationKit
using LinearAlgebra: mul!
"""
    LogitGame(pass!, target_precision)

A logit game whose equilibria solve `z = λ G(z, λ)` at precision `λ = 1 / temperature`, `G` the payoff gains the
logits scale. `pass!(gain, gain_slope, gain_precision_slope, z, λ)` writes `G(z, λ)`, its Jacobian `∂G/∂z` and its
slope `∂G/∂λ` into the three arrays, from one evaluation, and returns `nothing`. The arrays are views into the
trace's own residual and Jacobian (`BranchWorkspace`), so a pass allocates nothing of its own. A pass that needs work
arrays of its own gets them from `pass_workspace`. `G` must be finite at `λ = 0`, where the flat logits are the only
equilibrium. `target_precision` is the precision a trace ends at.
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
    pass_workspace(pass!, logits)

The work arrays `pass!` needs beyond its outputs, in the element type and length of `logits`, or `nothing` (the
default) when it needs none. Each `BranchWorkspace` builds its own, so a game shared by two traces stays safe. A pass
with work arrays is called as `pass!(work, gain, gain_slope, gain_precision_slope, z, λ)`.
"""
pass_workspace(::Any, ::AbstractVector)::Nothing = nothing

"""Call `pass!` on its outputs, with its work arrays `work` unless it has none."""
function run_pass!(
    pass!::Any,
    work::Any,
    gain::AbstractVector,
    gain_slope::AbstractMatrix,
    gain_precision_slope::AbstractVector,
    logits::AbstractVector,
    precision::Real,
)::Nothing
    if isnothing(work)
        pass!(gain, gain_slope, gain_precision_slope, logits, precision)
    else
        pass!(work, gain, gain_slope, gain_precision_slope, logits, precision)
    end
    return nothing
end

"""
    BranchWorkspace(game, point)

A residual, Jacobian, logits and `game`'s pass work arrays (`pass_workspace`) for branch points like
`point = [w; ρ]`, refilled by every call of the one callable that owns it. BifurcationKit consumes each result
before calling that callable again, so the fused system and the Jacobian each own one workspace per trace; a shared
one would let `J` overwrite a residual `FJ` returned.
"""
struct BranchWorkspace{E <: Real, W}
    residual::Vector{E}
    jacobian::Matrix{E}
    logits::Vector{E}
    pass_work::W
end

function BranchWorkspace(game::LogitGame, point::AbstractVector{E})::BranchWorkspace{E} where {E <: Real}
    return BranchWorkspace(
        Vector{E}(undef, length(point)),
        Matrix{E}(undef, length(point), length(point)),
        Vector{E}(undef, length(point) - 1),
        pass_workspace(game.pass!, state_of(point)),
    )
end

"""
    GameResidual(game)

BifurcationKit's `F(x, pin)` of `game`'s branch in payoff gaps, with the relative precision carried as the state's
last coordinate: `x = [w; ρ]`, `F = [w - P G(ρ w, ρ P); ρ - pin]`, `P` the target precision. The logits are `z = ρ w`,
so `F = 0` is `z = λ G(z, λ)` at `λ = ρ P`, and `F` stays regular at `ρ = 0`.
`pin` is BifurcationKit's continuation parameter, which only pins `ρ`. Each call fills a fresh `BranchWorkspace`:
BifurcationKit keeps its problem's `F` apart from the fused system.
"""
struct GameResidual{G <: LogitGame}
    game::G
end

function (residual::GameResidual)(point::AbstractVector{E}, pin::Real)::AbstractVector{E} where {E <: Real}
    return branch_system!(BranchWorkspace(residual.game, point), residual.game, point, pin).residual
end

"""
    GameSystem(game, workspace)

BifurcationKit's fused `FJ(x, pin)` of `game`'s branch: the residual and Jacobian from one `pass!`, written into
`workspace` (`branch_system!`). PALC's Newton method calls it in place of the residual and `J`.
"""
struct GameSystem{G <: LogitGame, W <: BranchWorkspace}
    game::G
    workspace::W
end

function (system::GameSystem)(point::AbstractVector{E}, pin::Real)::Tuple{
    Vector{E},
    Matrix{E},
} where {E <: Real}
    filled = branch_system!(system.workspace, system.game, point, pin)
    return filled.residual, filled.jacobian
end

"""BifurcationKit's `J(x, pin)` of `game`'s branch, written into `workspace`."""
struct GameJacobian{G <: LogitGame, W <: BranchWorkspace}
    game::G
    workspace::W
end

function (jacobian::GameJacobian)(point::AbstractVector{E}, pin::Real)::AbstractMatrix{E} where {E <: Real}
    return branch_system!(jacobian.workspace, jacobian.game, point, pin).jacobian
end

"""The payoff gaps `w` of a branch point `x = [w; ρ]`."""
state_of(point::AbstractVector)::AbstractVector = view(
    point,
    firstindex(point):(lastindex(point) - 1),
)

"""The logits `z = ρ w` of a branch point `x = [w; ρ]`."""
function logits_at(point::AbstractVector{E})::AbstractVector{E} where {E <: Real}
    return last(point) .* state_of(point)
end

"""
    branch_system!(workspace, game, x, pin)

`workspace` with `game`'s branch residual and Jacobian at `x = [w; ρ]`, `λ = ρ P`: `∂F/∂w = I - λ ∂G/∂z` and
`∂F/∂ρ = -P (∂G/∂z w + P ∂G/∂λ)`, over the relative precision's own row `[0 … 0 1]`. The pass writes `G`, `∂G/∂z`
and `∂G/∂λ` into the blocks that hold `F` and its Jacobian, which are then formed in place.
"""
function branch_system!(
    workspace::BranchWorkspace{E},
    game::LogitGame,
    point::AbstractVector{E},
    pin::Real,
)::typeof(workspace) where {E <: Real}
    (; residual, jacobian, logits, pass_work) = workspace
    gaps = state_of(point)
    width = length(gaps)
    relative_precision = point[end]
    logits .= relative_precision .* gaps
    precision = precision_at(game, relative_precision)
    target = game.target_precision
    gain = view(residual, 1:width)
    # fatou-ignore call-arity: Base `view` takes any number of indices (jolars/fatou#125)
    gain_slope = view(jacobian, 1:width, 1:width)
    # fatou-ignore call-arity: Base `view` takes any number of indices (jolars/fatou#125)
    precision_column = view(jacobian, 1:width, width + 1)
    run_pass!(game.pass!, pass_work, gain, gain_slope, precision_column, logits, precision)
    # `∂F/∂ρ` first: it reads `∂G/∂z`, which the next line overwrites.
    mul!(precision_column, gain_slope, gaps, -target, -target^2)
    gain_slope .= .-precision .* gain_slope
    for diagonal in axes(gain_slope, 1)
        gain_slope[diagonal, diagonal] += one(E)
    end
    gain .= gaps .- target .* gain
    # fatou-ignore call-arity: Base `view` takes any number of indices (jolars/fatou#125)
    fill!(view(jacobian, width + 1, 1:width), zero(E))
    jacobian[end, end] = one(E)
    residual[end] = relative_precision - pin
    return workspace
end

"""
    trace_start(game, flat)

The branch's start `[w; 0]` at zero precision: the gaps `P G(0, 0)` that the `flat` logits (all zero) face.
"""
function trace_start(game::LogitGame, flat::AbstractVector{E})::AbstractVector{E} where {E <: Real}
    start = [flat; zero(E)]
    state_of(start) .= .-state_of(branch_system!(BranchWorkspace(game, start), game, start, zero(E)).residual)
    return start
end

"""
    PinSlope(start)

`∂F/∂pin` of the branch residual `[F(w, ρ); ρ - pin]`, BifurcationKit's jet `R01`: `[0 … 0 -1]`, exactly, in
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
    branch_problem(game, start, pin; record_from_solution)

BifurcationKit's problem for `game`'s branch from `start = [w; ρ]` at `pin`, with the fused system and
exact `∂F/∂pin` supplied. The system and Jacobian each own a `BranchWorkspace`.
"""
function branch_problem(
    game::LogitGame,
    start::AbstractVector{E},
    pin::Real;
    record_from_solution::Any,
)::BifurcationKit.AbstractBifurcationProblem where {E <: Real}
    return BifurcationKit.BifurcationProblem(
        GameResidual(game),
        start,
        pin;
        J = GameJacobian(game, BranchWorkspace(game, start)),
        FJ = GameSystem(game, BranchWorkspace(game, start)),
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
