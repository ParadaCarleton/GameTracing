"""
How a trace's state coordinates `u` map to logits `z`: the `coordinates` setting of `TraceSettings`.
`AsinhLogits` (`z = sinh(u)`, the default) grows `u` like `log z`, so a logit that grows with the precision keeps
an arclength commensurate with the relative precision's. `PlainLogits` (`z = u`) traces the logits themselves.
"""
abstract type LogitCoordinates end

"""State coordinates `u = asinh(z)`: logarithmic in large logits, linear near zero."""
struct AsinhLogits <: LogitCoordinates end

"""State coordinates equal to the logits."""
struct PlainLogits <: LogitCoordinates end

"""The logit at state coordinate `coordinate`."""
function logit_at(::AsinhLogits, coordinate::R)::R where {R <: Real}
    return sinh(coordinate)
end

function logit_at(::PlainLogits, coordinate::R)::R where {R <: Real}
    return coordinate
end

"""`dz/du`, the logit's rate of change in its state coordinate, at `coordinate`."""
function logit_rate(::AsinhLogits, coordinate::R)::R where {R <: Real}
    return cosh(coordinate)
end

function logit_rate(::PlainLogits, coordinate::R)::R where {R <: Real}
    return one(coordinate)
end

"""The state coordinate of `logit`."""
function state_at(::AsinhLogits, logit::R)::R where {R <: Real}
    return asinh(logit)
end

function state_at(::PlainLogits, logit::R)::R where {R <: Real}
    return logit
end

"""The logits at state coordinates `state`."""
function logits_at(coordinates::LogitCoordinates, state::AbstractVector{R})::AbstractVector{R} where {R <: Real}
    return logit_at.((coordinates,), state)
end

"""The state coordinates of `logits`."""
function states_at(coordinates::LogitCoordinates, logits::AbstractVector{R})::AbstractVector{R} where {R <: Real}
    return state_at.((coordinates,), logits)
end
