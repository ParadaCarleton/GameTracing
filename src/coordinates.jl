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
logit_at(::AsinhLogits, u::R)::R where {R <: Real} = sinh(u)
logit_at(::PlainLogits, u::R)::R where {R <: Real} = u

"""`dz/du`, the logit's rate of change in its state coordinate `u`."""
logit_rate(::AsinhLogits, u::R)::R where {R <: Real} = cosh(u)
logit_rate(::PlainLogits, u::R)::R where {R <: Real} = one(u)

"""The state coordinate of logit `z`."""
state_at(::AsinhLogits, z::R)::R where {R <: Real} = asinh(z)
state_at(::PlainLogits, z::R)::R where {R <: Real} = z

"""The logits at state coordinates `state`."""
logits_at(coordinates::LogitCoordinates, state::AbstractVector{R})::AbstractVector{R} where {R <: Real} = [
    logit_at(coordinates, entry) for entry in state
]

"""The state coordinates of `logits`."""
states_at(coordinates::LogitCoordinates, logits::AbstractVector{R})::AbstractVector{R} where {R <: Real} = [
    state_at(coordinates, entry) for entry in logits
]
