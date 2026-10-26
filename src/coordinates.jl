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

"""The logit at state coordinate `u`."""
function logit_at(::AsinhLogits, u::R)::R where {R <: Real}
    return sinh(u)
end

function logit_at(::PlainLogits, u::R)::R where {R <: Real}
    return u
end

"""`dz/du`, the logit's rate of change in its state coordinate `u`."""
function logit_rate(::AsinhLogits, u::R)::R where {R <: Real}
    return cosh(u)
end

function logit_rate(::PlainLogits, u::R)::R where {R <: Real}
    return one(u)
end

"""The state coordinate of logit `z`."""
function state_at(::AsinhLogits, z::R)::R where {R <: Real}
    return asinh(z)
end

function state_at(::PlainLogits, z::R)::R where {R <: Real}
    return z
end

"""The logits at state coordinates `state`."""
function logits_at(coordinates::LogitCoordinates, state::AbstractVector{R})::AbstractVector{R} where {R <: Real}
    return [logit_at(coordinates, entry) for entry in state]
end

"""The state coordinates of `logits`."""
function states_at(coordinates::LogitCoordinates, logits::AbstractVector{R})::AbstractVector{R} where {R <: Real}
    return [state_at(coordinates, entry) for entry in logits]
end
