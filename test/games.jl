using Random: RandomDevice, Xoshiro

"""A generator on a fresh seed, logged so a failing draw can be replayed with `Xoshiro(seed)`."""
function fresh_rng()::Xoshiro
    seed = rand(RandomDevice(), UInt)
    @info "fresh draw" seed
    return Xoshiro(seed)
end

"""Logit response `logits = precision (A p + b)` with `p = logistic(logits)`: payoff gains `A p + b`, `T` zero at precision zero."""
struct LinearGainPass{E <: Real}
    interaction::Matrix{E}
    stand_alone::Vector{E}
end

function (pass::LinearGainPass)(
    response::AbstractVector,
    logit_slope::AbstractMatrix,
    precision_slope::AbstractVector,
    logits::AbstractVector{E},
    precision::Real,
)::Nothing where {E <: Real}
    stays = inv.(one(E) .+ exp.(-logits))
    precision_slope .= pass.interaction * stays .+ pass.stand_alone
    response .= precision .* precision_slope
    logit_slope .= precision .* pass.interaction .* transpose(stays .* (one(E) .- stays))
    return nothing
end

"""One logit that gains `amplitude * logistic(logits - threshold)`: a coordination game whose branch folds back when `threshold` is large."""
struct ThresholdPass{E <: Real}
    amplitude::E
    threshold::E
end

function (pass::ThresholdPass)(
    response::AbstractVector,
    logit_slope::AbstractMatrix,
    precision_slope::AbstractVector,
    logits::AbstractVector{E},
    precision::Real,
)::Nothing where {E <: Real}
    stays = inv.(one(E) .+ exp.(pass.threshold .- logits))
    precision_slope .= pass.amplitude .* stays
    response .= precision .* precision_slope
    logit_slope .= reshape(precision .* pass.amplitude .* stays .* (one(E) .- stays), 1, 1)
    return nothing
end

"""A `LinearGainPass` whose response is NaN once the precision passes `poison`."""
struct PoisonedPass{P, E <: Real}
    pass::P
    poison::E
end

function (pass::PoisonedPass)(
    response::AbstractVector,
    logit_slope::AbstractMatrix,
    precision_slope::AbstractVector,
    logits::AbstractVector,
    precision::Real,
)::Nothing
    pass.pass(response, logit_slope, precision_slope, logits, precision)
    if precision > pass.poison
        fill!(response, NaN32)
    end
    return nothing
end

"""A pass that gives a NaN slope, so the Jacobian is non-finite where the residual is not."""
struct NaNSlopePass{P}
    pass::P
end

function (pass::NaNSlopePass)(
    response::AbstractVector,
    logit_slope::AbstractMatrix,
    precision_slope::AbstractVector,
    logits::AbstractVector,
    precision::Real,
)::Nothing
    pass.pass(response, logit_slope, precision_slope, logits, precision)
    fill!(logit_slope, NaN32)
    return nothing
end

"""A pass that throws a caller's `ArgumentError` once the precision is positive."""
struct ThrowingPass{P}
    pass::P
end

function (pass::ThrowingPass)(
    response::AbstractVector,
    logit_slope::AbstractMatrix,
    precision_slope::AbstractVector,
    logits::AbstractVector,
    precision::Real,
)::Nothing
    if precision > 0
        throw(ArgumentError("bad game"))
    end
    pass.pass(response, logit_slope, precision_slope, logits, precision)
    return nothing
end

"""`(; response, logit_slope, precision_slope)` of `pass` at `logits` and `precision`, in fresh arrays."""
function evaluated(pass::Any, logits::AbstractVector{E}, precision::Real)::NamedTuple where {E <: Real}
    width = length(logits)
    response = Vector{E}(undef, width)
    logit_slope = Matrix{E}(undef, width, width)
    precision_slope = Vector{E}(undef, width)
    pass(response, logit_slope, precision_slope, logits, precision)
    return (; response, logit_slope, precision_slope)
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
Logit response of a normal-form game in which every player has two actions: `z_p = precision (U_p(1, π) - U_p(2, π))`,
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
    response::AbstractVector,
    logit_slope::AbstractMatrix,
    precision_slope::AbstractVector,
    logits::AbstractVector{E},
    precision::Real,
)::Nothing where {N, E <: Real}
    first_choices = inv.(one(E) .+ exp.(-logits))
    fill!(precision_slope, zero(E))
    fill!(logit_slope, zero(E))
    for profile in CartesianIndices(first(pass.payoffs)), (player, payoff) in pairs(pass.payoffs)
        signed = action_sign(profile[player]) * payoff[profile]
        chances = [action_probability(first_choices[rival], profile[rival]) for rival in 1:N]
        precision_slope[player] += signed * prod((chances[rival] for rival in 1:N if rival != player); init = one(E))
        for rival in 1:N
            if rival != player
                others = prod((chances[other] for other in 1:N if other != player && other != rival); init = one(E))
                logit_slope[player, rival] += signed * action_sign(profile[rival]) * others
            end
        end
    end
    logit_slope .*= precision .* transpose(first_choices .* (one(E) .- first_choices))
    response .= precision .* precision_slope
    return nothing
end

"""
    branch_reference(pass, width, precision)

`(; reached, logits, first_choices)`: an independent reference for the principal branch of `pass` at `precision`, by natural
continuation in `Float64` from zero precision. Each step runs Newton from the last point and halves until Newton
converges with no logit moving more than `1/1000`, so it creeps through the sharp turn of an unfolded pitchfork. A
branch that folds back in precision has no next point, and the reference stops short (`reached = false`).
"""
function branch_reference(pass::Any, width::Integer, precision::Real)::NamedTuple
    logits = zeros(width)
    # `float(1)` widens a 32-bit precision, which no Float64 literal names.
    target = precision * float(1)
    level, step = zero(target), inv(float(1000))
    while level < target && step > eps(target)
        next = min(level + step, target)
        candidate = copy(logits)
        for _ in 1:4
            gains = evaluated(pass, candidate, next)
            candidate = candidate .- (one(gains.logit_slope) .- gains.logit_slope) \ (candidate .- gains.response)
        end
        settled = residual_norm(pass, candidate, next) <= 10^4 * eps(target) * (1 + maximum(abs, candidate))
        if settled && maximum(abs, candidate .- logits) <= inv(float(1000))
            logits, level, step = candidate, next, min(2 * step, inv(float(100)))
        else
            step /= 2
        end
    end
    return (; reached = level == target, logits, first_choices = inv.(1 .+ exp.(-logits)))
end
