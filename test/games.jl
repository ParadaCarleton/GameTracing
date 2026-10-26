using Random: RandomDevice, Xoshiro

"""A generator on a fresh seed, logged so a failing draw can be replayed with `Xoshiro(seed)`."""
function fresh_rng()::Xoshiro
    seed = rand(RandomDevice(), UInt)
    @info "fresh draw" seed
    return Xoshiro(seed)
end

"""Payoff gains `G = A p + b` with `p = logistic(logits)`, independent of the precision: the logit map `λ (A p + b)`."""
struct LinearGainPass{E <: Real}
    interaction::Matrix{E}
    stand_alone::Vector{E}
end

function (pass::LinearGainPass)(
    gain::AbstractVector,
    gain_slope::AbstractMatrix,
    gain_precision_slope::AbstractVector,
    logits::AbstractVector{E},
    ::Real,
)::Nothing where {E <: Real}
    stays = inv.(one(E) .+ exp.(-logits))
    gain .= pass.interaction * stays .+ pass.stand_alone
    gain_slope .= pass.interaction .* transpose(stays .* (one(E) .- stays))
    fill!(gain_precision_slope, zero(E))
    return nothing
end

"""One logit that gains `amplitude * logistic(logits - threshold)`: a coordination game whose branch folds back when `threshold` is large."""
struct ThresholdPass{E <: Real}
    amplitude::E
    threshold::E
end

function (pass::ThresholdPass)(
    gain::AbstractVector,
    gain_slope::AbstractMatrix,
    gain_precision_slope::AbstractVector,
    logits::AbstractVector{E},
    ::Real,
)::Nothing where {E <: Real}
    stays = inv.(one(E) .+ exp.(pass.threshold .- logits))
    gain .= pass.amplitude .* stays
    gain_slope .= reshape(pass.amplitude .* stays .* (one(E) .- stays), 1, 1)
    fill!(gain_precision_slope, zero(E))
    return nothing
end

"""A `LinearGainPass` whose gains shrink with the precision: `G = (A p + b) / sqrt(1 + (λ s)²)`, so `∂G/∂λ ≠ 0`."""
struct DampedGainPass{E <: Real}
    pass::LinearGainPass{E}
    spread::E
end

function (pass::DampedGainPass)(
    gain::AbstractVector,
    gain_slope::AbstractMatrix,
    gain_precision_slope::AbstractVector,
    logits::AbstractVector{E},
    precision::Real,
)::Nothing where {E <: Real}
    pass.pass(gain, gain_slope, gain_precision_slope, logits, precision)
    damping = inv(sqrt(1 + (precision * pass.spread)^2))
    gain_precision_slope .= -precision * pass.spread^2 * damping^2 .* damping .* gain
    gain .*= damping
    gain_slope .*= damping
    return nothing
end

"""One logit whose gain is the precision itself, `G = λ`: the branch `z = λ²`."""
struct PrecisionGainPass end

function (::PrecisionGainPass)(
    gain::AbstractVector,
    gain_slope::AbstractMatrix,
    gain_precision_slope::AbstractVector,
    ::AbstractVector{E},
    precision::Real,
)::Nothing where {E <: Real}
    fill!(gain, E(precision))
    fill!(gain_slope, zero(E))
    fill!(gain_precision_slope, one(E))
    return nothing
end

"""A pass whose gain is NaN once the precision passes `poison`."""
struct PoisonedPass{P, E <: Real}
    pass::P
    poison::E
end

function (pass::PoisonedPass)(
    gain::AbstractVector,
    gain_slope::AbstractMatrix,
    gain_precision_slope::AbstractVector,
    logits::AbstractVector,
    precision::Real,
)::Nothing
    pass.pass(gain, gain_slope, gain_precision_slope, logits, precision)
    if precision > pass.poison
        fill!(gain, NaN32)
    end
    return nothing
end

"""A pass that gives a NaN slope, so the Jacobian is non-finite where the residual is not."""
struct NaNSlopePass{P}
    pass::P
end

function (pass::NaNSlopePass)(
    gain::AbstractVector,
    gain_slope::AbstractMatrix,
    gain_precision_slope::AbstractVector,
    logits::AbstractVector,
    precision::Real,
)::Nothing
    pass.pass(gain, gain_slope, gain_precision_slope, logits, precision)
    fill!(gain_slope, NaN32)
    return nothing
end

"""A pass that throws a caller's `ArgumentError` once the precision is positive."""
struct ThrowingPass{P}
    pass::P
end

function (pass::ThrowingPass)(
    gain::AbstractVector,
    gain_slope::AbstractMatrix,
    gain_precision_slope::AbstractVector,
    logits::AbstractVector,
    precision::Real,
)::Nothing
    if precision > 0
        throw(ArgumentError("bad game"))
    end
    pass.pass(gain, gain_slope, gain_precision_slope, logits, precision)
    return nothing
end

"""
The logit map `T = λ G` of `pass` at `logits` and precision `λ`, in fresh arrays: `(; response, logit_slope,
precision_slope)`, which are `T`, `∂T/∂z = λ ∂G/∂z` and `∂T/∂λ = G + λ ∂G/∂λ`.
"""
function evaluated(pass::Any, logits::AbstractVector{E}, precision::Real)::NamedTuple where {E <: Real}
    width = length(logits)
    gain = Vector{E}(undef, width)
    gain_slope = Matrix{E}(undef, width, width)
    gain_precision_slope = Vector{E}(undef, width)
    pass(gain, gain_slope, gain_precision_slope, logits, precision)
    return (;
        response = precision .* gain,
        logit_slope = precision .* gain_slope,
        precision_slope = gain .+ precision .* gain_precision_slope,
    )
end

"""The largest `|logits - T(logits, precision)|` entry: zero at an equilibrium."""
residual_norm(pass::Any, logits::AbstractVector, precision::Real) = maximum(
    abs,
    logits .- evaluated(pass, logits, precision).response,
)

"""Picard iterates of `logits -> T(logits, precision)` from the flat start: the equilibrium when `T` contracts."""
function picard_equilibrium(
    pass::Any,
    width::Integer,
    precision::E,
    rounds::Integer,
)::Vector{E} where {E <: Real}
    logits = zeros(E, width)
    for _ in 1:rounds
        logits = evaluated(pass, logits, precision).response
    end
    return logits
end

"""
Payoff gains of a normal-form game in which every player has two actions: `G_p = U_p(1, π) - U_p(2, π)`,
each rival `q` playing its first action with probability `logistic(z_q)`. `payoffs[p][profile]` is player `p`'s payoff.
"""
struct NormalFormPass{N, A <: AbstractArray{<:Real, N}}
    payoffs::NTuple{N, A}
end

"""The probability that a player whose first action has probability `first_choice` plays `action`."""
function action_probability(first_choice::E, action::Integer)::E where {E <: Real}
    return ifelse(action == 1, first_choice, 1 - first_choice)
end

"""`+1` for a player's first action, `-1` for its second: the sign it carries in a payoff gain."""
action_sign(action::Integer)::Int = 3 - 2 * action

function (pass::NormalFormPass{N})(
    gain::AbstractVector,
    gain_slope::AbstractMatrix,
    gain_precision_slope::AbstractVector,
    logits::AbstractVector{E},
    ::Real,
)::Nothing where {N, E <: Real}
    first_choices = inv.(one(E) .+ exp.(-logits))
    fill!(gain, zero(E))
    fill!(gain_slope, zero(E))
    fill!(gain_precision_slope, zero(E))
    for profile in CartesianIndices(first(pass.payoffs)), (player, payoff) in pairs(pass.payoffs)
        signed = action_sign(profile[player]) * payoff[profile]
        chances = [action_probability(first_choices[rival], profile[rival]) for rival in 1:N]
        gain[player] += signed * prod((chances[rival] for rival in 1:N if rival != player); init = one(E))
        for rival in 1:N
            if rival != player
                others = prod((chances[other] for other in 1:N if other != player && other != rival); init = one(E))
                gain_slope[player, rival] += signed * action_sign(profile[rival]) * others
            end
        end
    end
    gain_slope .*= transpose(first_choices .* (one(E) .- first_choices))
    return nothing
end

"""Whether `logits` solves `pass` at `precision` to `Float64` rounding at `scale`, times the logits' size."""
function settled_root(pass::Any, logits::AbstractVector{E}, precision::E, scale::E)::Bool where {E <: Real}
    return residual_norm(pass, logits, precision) <= 10^4 * eps(scale) * (1 + maximum(abs, logits))
end

"""
    level_root(pass, logits, precision)

The root of `pass` at fixed `precision` that `Float64` Newton reaches from `logits`; an `ErrorException` when 50 steps do not settle.
"""
function level_root(pass::Any, logits::AbstractVector, precision::Real)::Vector{Float64}
    root, level = float.(collect(logits)) .* float(1), precision * float(1)
    for _ in 1:50
        if settled_root(pass, root, level, max(level, one(level)))
            return root
        end
        gains = evaluated(pass, root, level)
        root = root .- (one(gains.logit_slope) .- gains.logit_slope) \ (root .- gains.response)
    end
    error("level_root: Newton did not settle at precision $level")
end

"""
    branch_reference(pass, width, precision; start, start_precision)

`(; reached, logits, first_choices)`: an independent reference for the branch of `pass` at `precision`, by natural
continuation in `Float64` from the root `level_root` reaches from `start` at `start_precision` (default: zero logits at
zero precision, the principal branch). Each step runs Newton from the last point and halves until Newton converges with
no logit moving more than `1/1000`, so it creeps through the sharp turn of an unfolded pitchfork. A branch that folds
back in precision has no next point, and the reference stops short (`reached = false`).
"""
function branch_reference(
    pass::Any,
    width::Integer,
    precision::Real;
    start::AbstractVector = zeros(width),
    start_precision::Real = 0,
)::NamedTuple
    logits = level_root(pass, start, start_precision)
    # `float(1)` widens a 32-bit precision, which no Float64 literal names.
    target = precision * float(1)
    level, step = start_precision * float(1), inv(float(1000))
    while level < target && step > eps(target)
        next = min(level + step, target)
        candidate = copy(logits)
        for _ in 1:4
            gains = evaluated(pass, candidate, next)
            candidate = candidate .- (one(gains.logit_slope) .- gains.logit_slope) \ (candidate .- gains.response)
        end
        if settled_root(pass, candidate, next, target) && maximum(abs, candidate .- logits) <= inv(float(1000))
            logits, level, step = candidate, next, min(2 * step, inv(float(100)))
        else
            step /= 2
        end
    end
    return (; reached = level == target, logits, first_choices = inv.(1 .+ exp.(-logits)))
end

"""
    MapSwap(before, after, precision, swap_level, width)

A trace over `width` logits to `precision` whose game changes from `before` to `after` once the trace reaches relative
precision `swap_level`: a redraw of the map at an accepted point.
"""
struct MapSwap{B, A, E <: Real}
    before::B
    after::A
    precision::E
    swap_level::E
    width::Int
end

# The swap moves a two-player branch by under a tenth: a redraw a fixed-precision correction absorbs.
const NEAR_SWAP = MapSwap(
    LinearGainPass(Float32[0 0.8; 0.6 0], Float32[0.3, -0.2]),
    LinearGainPass(Float32[0 0.8; 0.6 0], Float32[0.34, -0.23]),
    4.0f0, 0.5f0, 2,
)

# The swap removes the low branch at precision 1/2: the only root there is the high one, about 3.5 logits away.
const FAR_SWAP = MapSwap(ThresholdPass(8.0f0, 4.0f0), ThresholdPass(8.0f0, 1.5f0), 1.0f0, 0.5f0, 1)

"""
    SwapAt(game, swap_level)

A `redraw` hook that returns `game` at every accepted point from relative precision `swap_level` on, and `nothing`
before: the trace swaps once, since returning its current game swaps nothing.
"""
struct SwapAt{G, E <: Real}
    game::G
    swap_level::E
end

function (hook::SwapAt{G})(::AbstractVector, relative_precision::Real)::Union{Nothing, G} where {G}
    if relative_precision >= hook.swap_level
        return hook.game
    end
    return nothing
end

"""A `keep_going` hook that stops the trace at its first accepted point at or past `swap_level`."""
struct StopAt{E <: Real}
    swap_level::E
end

(hook::StopAt)(::AbstractVector, relative_precision::Real)::Bool = relative_precision < hook.swap_level
