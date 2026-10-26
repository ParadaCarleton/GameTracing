"""Logit response `logits = precision (A p + b)` with `p = logistic(logits)`: payoff gains `A p + b`, `T` zero at precision zero."""
struct LinearGainPass{E <: Real}
    interaction::Matrix{E}
    stand_alone::Vector{E}
end

function (pass::LinearGainPass)(logits::AbstractVector{E}, precision::Real) where {E <:
    Real}
    stays = inv.(one(E) .+ exp.(-logits))
    gains = pass.interaction * stays .+ pass.stand_alone
    slopes = stays .* (one(E) .- stays)
    return (;
        response = precision .* gains,
        logit_slope = precision .* pass.interaction .* transpose(slopes),
        precision_slope = gains,
    )
end

"""One logit that gains `amplitude * logistic(logits - threshold)`: a coordination game whose branch folds back when `threshold` is large."""
struct ThresholdPass{E <: Real}
    amplitude::E
    threshold::E
end

function (pass::ThresholdPass)(logits::AbstractVector{E}, precision::Real) where {E <: Real}
    stays = inv.(one(E) .+ exp.(pass.threshold .- logits))
    gains = pass.amplitude .* stays
    return (;
        response = precision .* gains,
        logit_slope = reshape(
            precision .* pass.amplitude .* stays .* (one(E) .- stays),
            1,
            1,
        ),
        precision_slope = gains,
    )
end

"""A `LinearGainPass` whose response is NaN once the precision passes `poison`."""
struct PoisonedPass{P, E <: Real}
    pass::P
    poison::E
end

function (pass::PoisonedPass)(logits::AbstractVector, precision::Real)
    gains = pass.pass(logits, precision)
    if precision > pass.poison
        return (; gains..., response = fill(oftype(precision, NaN32), length(logits)))
    end
    return gains
end

"""A pass that gives a NaN slope, so the Jacobian is non-finite where the residual is not."""
struct NaNSlopePass{P}
    pass::P
end

function (pass::NaNSlopePass)(logits::AbstractVector, precision::Real)
    gains = pass.pass(logits, precision)
    return (; gains..., logit_slope = fill(
        oftype(precision, NaN32),
        length(logits),
        length(logits),
    ))
end

"""A pass that throws a caller's `ArgumentError` once the precision is positive."""
struct ThrowingPass{P}
    pass::P
end

function (pass::ThrowingPass)(logits::AbstractVector, precision::Real)
    if precision > 0
        throw(ArgumentError("bad game"))
    end
    return pass.pass(logits, precision)
end

"""The largest `|logits - T(logits, precision)|` entry: zero at an equilibrium."""
residual_norm(pass::Any, logits::AbstractVector, precision::Real) = maximum(
    abs,
    logits .- pass(logits, precision).response,
)

"""
Logit response of a normal-form game in which every player has two actions: `z_p = precision (U_p(1, π) - U_p(2, π))`,
each rival `q` playing its first action with probability `logistic(z_q)`. `payoffs[p][profile]` is player `p`'s payoff.
"""
struct NormalFormPass{N, E <: Real}
    payoffs::NTuple{N, Array{E, N}}
end

"""The probability that a player whose first action has probability `first_choice` plays `action`."""
function action_probability(first_choice::E, action::Integer)::E where {E <: Real}
    return ifelse(action == 1, first_choice, 1 - first_choice)
end

"""`+1` for a player's first action, `-1` for its second: the sign it carries in a payoff gain."""
action_sign(action::Integer)::Int = 3 - 2 * action

function (pass::NormalFormPass{N, E})(logits::AbstractVector{E}, precision::Real) where {N, E <: Real}
    first_choices = inv.(one(E) .+ exp.(-logits))
    gains = zeros(E, N)
    gain_slopes = zeros(E, N, N)
    for profile in CartesianIndices(first(pass.payoffs)), (player, payoff) in pairs(pass.payoffs)
        signed = action_sign(profile[player]) * payoff[profile]
        chances = [action_probability(first_choices[rival], profile[rival]) for rival in 1:N]
        gains[player] += signed * prod((chances[rival] for rival in 1:N if rival != player); init = one(E))
        for rival in 1:N
            if rival != player
                others = prod((chances[other] for other in 1:N if other != player && other != rival); init = one(E))
                gain_slopes[player, rival] += signed * action_sign(profile[rival]) * others
            end
        end
    end
    return (;
        response = precision .* gains,
        logit_slope = precision .* gain_slopes .* transpose(first_choices .* (one(E) .- first_choices)),
        precision_slope = gains,
    )
end

"""Picard iterates of `logits -> T(logits, precision)` from the flat start: the equilibrium when `T` contracts."""
function picard_equilibrium(
    pass::Any,
    width::Integer,
    precision::E,
    rounds::Integer,
)::Vector{E} where {E <: Real}
    logits = zeros(E, width)
    for _ in 1:rounds
        logits = pass(logits, precision).response
    end
    return logits
end
