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
