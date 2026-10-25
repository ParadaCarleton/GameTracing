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
