using BifurcationKit: BifurcationKit
using LinearAlgebra: dot
using LogExpFunctions: log1pexp, logistic
"""
The trace's state coordinates `x` for logits `z` at relative precision `ρ`: `AsinhLogits` (`x = asinh(z)`), `Logits`
(`x = z`), or `PayoffGaps` (`x = z / ρ`, the payoff gain the logit scales, which stays finite at `ρ = 0`).
"""
@enum TraceCoordinates AsinhLogits Logits PayoffGaps

"""
The trace's step length, in a state `x = [u; ρ]` of `k` coordinates `u` and state weight `θ`: `AveragedDistance` is
`θ ‖Δu‖² / k + (1 - θ) Δρ²`, `SummedDistance` `θ ‖Δu‖² + Δρ²`, and `FisherDistance` `θ Σ σ(zᵢ)σ(-zᵢ) Δuᵢ² + Δρ²`
with the Bernoulli Fisher weights of the logits `z` frozen at each accepted point. The reach and angle checks of a
`FisherDistance` trace measure in the summed norm.
"""
@enum StepDistance AveragedDistance SummedDistance FisherDistance

"""
    TraceGeometry{E}(; coordinates, distance, state_weight)

The space a trace steps in: its `TraceCoordinates`, its `StepDistance`, and `state_weight`, the coordinates' weight
`θ` against the relative precision's in that distance. The default is the asinh trace in PALC's averaged norm.
"""
Base.@kwdef struct TraceGeometry{E <: Real}
    coordinates::TraceCoordinates = AsinhLogits
    distance::StepDistance = AveragedDistance
    state_weight::E = 0.5f0
end

"""`dz/du` of a logit `z` at its trace coordinate `coordinate`: `cosh` for `AsinhLogits`, else 1 (payoff gaps fold `ρ` into `λ`)."""
function logit_scale(coordinates::TraceCoordinates, coordinate::E)::E where {E <: Real}
    return ifelse(coordinates == AsinhLogits, cosh(coordinate), one(E))
end

"""`logits` filled with the logits of trace state `state` in `coordinates` at relative precision `relative_precision`."""
function trace_logits!(
    logits::AbstractVector{E},
    coordinates::TraceCoordinates,
    state::AbstractVector{E},
    relative_precision::Real,
)::typeof(logits) where {E <: Real}
    if coordinates == AsinhLogits
        return map!(sinh, logits, state)
    elseif coordinates == Logits
        return copyto!(logits, state)
    end
    logits .= relative_precision .* state
    return logits
end

"""The logits of trace state `state` in `coordinates` at relative precision `relative_precision`, in a fresh vector."""
function trace_logits(
    coordinates::TraceCoordinates,
    state::AbstractVector{E},
    relative_precision::Real,
)::Vector{E} where {E <: Real}
    return trace_logits!(similar(state, E), coordinates, state, relative_precision)
end

"""The trace state of `logits` in `coordinates` at a positive relative precision (payoff gaps have none at zero)."""
function trace_state(
    coordinates::TraceCoordinates,
    logits::AbstractVector{E},
    relative_precision::Real,
)::Vector{E} where {E <: Real}
    if coordinates == AsinhLogits
        return asinh.(logits)
    elseif coordinates == Logits
        return copy(logits)
    end
    if !(relative_precision > 0)
        throw(ArgumentError("payoff gaps have no state at relative precision $relative_precision"))
    end
    return logits ./ relative_precision
end

"""
The logit tangent `dz` of a branch tangent `(state_tangent, precision_tangent)` at `state` and `relative_precision`:
`cosh(x) dx`, `dx`, or `ρ dx + x dρ` for payoff gaps.
"""
function logit_tangent(
    coordinates::TraceCoordinates,
    state::AbstractVector{E},
    relative_precision::Real,
    state_tangent::AbstractVector{E},
    precision_tangent::Real,
)::Vector{E} where {E <: Real}
    if coordinates == AsinhLogits
        return cosh.(state) .* state_tangent
    elseif coordinates == Logits
        return copy(state_tangent)
    end
    return relative_precision .* state_tangent .+ precision_tangent .* state
end

"""
The largest operand of the residual `F` at `state`, whose rounding floors the corrector's tolerance: the largest
logit, or the largest payoff gap for `PayoffGaps`, where `F = x - P G` is in gap units.
"""
function residual_scale(
    coordinates::TraceCoordinates,
    state::AbstractVector{E},
    relative_precision::Real,
)::E where {E <: Real}
    if coordinates == PayoffGaps
        return maximum(abs, state)
    end
    return maximum(abs, trace_logits(coordinates, state, relative_precision))
end

"""The weight of each `Δuᵢ²` in `geometry`'s plain norm over `width` coordinates (`StepDistance`)."""
function coordinate_weight(geometry::TraceGeometry{E}, width::Integer)::E where {E <: Real}
    if geometry.distance == AveragedDistance
        return geometry.state_weight / width
    end
    return geometry.state_weight
end

"""The weight of `Δρ²` in `geometry`'s norm (`StepDistance`)."""
function level_weight(geometry::TraceGeometry{E})::E where {E <: Real}
    if geometry.distance == AveragedDistance
        return one(E) - geometry.state_weight
    end
    return one(E)
end

"""The plain inner product of two displacements `[Δu; Δρ]` in `geometry` (`coordinate_weight`, `level_weight`)."""
function trace_dot(
    left::AbstractVector{E},
    right::AbstractVector{E},
    geometry::TraceGeometry{E},
)::E where {E <: Real}
    return coordinate_weight(geometry, length(left) - 1) * dot(state_of(left), state_of(right)) +
        level_weight(geometry) * last(left) * last(right)
end

"""The distance from `from` to `to`, both `[u; ρ]`, in `geometry`'s plain norm (`trace_dot`)."""
function arclength_distance(
    from::AbstractVector{E},
    to::AbstractVector{E},
    geometry::TraceGeometry{E},
)::E where {E <: Real}
    displacement = to .- from
    return sqrt(trace_dot(displacement, displacement, geometry))
end

"""
    chord_angle(previous, predicted, corrected, geometry)

The angle in radians between the predictor's direction `predicted - previous` and the chord `corrected - previous`,
in `geometry`'s plain inner product; zero when either is zero.
"""
function chord_angle(
    previous::AbstractVector{E},
    predicted::AbstractVector{E},
    corrected::AbstractVector{E},
    geometry::TraceGeometry{E},
)::E where {E <: Real}
    direction = predicted .- previous
    chord = corrected .- previous
    lengths = sqrt(trace_dot(direction, direction, geometry) * trace_dot(chord, chord, geometry))
    if !(lengths > 0)
        return zero(E)
    end
    return acos(clamp(trace_dot(direction, chord, geometry) / lengths, -one(E), one(E)))
end

"""`KL(Bernoulli(logistic(from)) ‖ Bernoulli(logistic(to)))`, from the logits, exact where a probability rounds to 0 or 1."""
function bernoulli_kl(from::E, to::E)::E where {E <: Real}
    probability = logistic(from)
    return probability * (log1pexp(-to) - log1pexp(-from)) + (one(E) - probability) * (log1pexp(to) - log1pexp(from))
end

"""The largest `bernoulli_kl` over the logits of two points `[u; ρ]` in `coordinates`."""
function largest_kl(
    coordinates::TraceCoordinates,
    from::AbstractVector{E},
    to::AbstractVector{E},
)::E where {E <: Real}
    from_logits = trace_logits(coordinates, state_of(from), last(from))
    to_logits = trace_logits(coordinates, state_of(to), last(to))
    return maximum(bernoulli_kl(start, finish) for (start, finish) in zip(from_logits, to_logits))
end

"""
    FisherWeights(coordinates)

`weights!` of a `FisherDistance` trace's `PointWeightedDot`: each state coordinate's Bernoulli Fisher weight
`σ(z)σ(-z)` at the accepted point's logits `z`, and one for the relative precision.
"""
struct FisherWeights
    coordinates::TraceCoordinates
end

function (fisher::FisherWeights)(
    weights::AbstractVector{E},
    point::BifurcationKit.BorderedArray,
)::AbstractVector{E} where {E <: Real}
    logits = trace_logits(fisher.coordinates, state_of(point.u), last(point.u))
    state_of(weights) .= logistic.(logits) .* logistic.(.-logits)
    weights[end] = one(E)
    return weights
end

"""
PALC's `θ` and `dotθ` for `geometry` over a trace of `width` coordinates: `augmented_theta` and BifurcationKit's
averaged `DotTheta()`, or the state weight itself with the plain sum `dot` or the Fisher `PointWeightedDot`, whose
`θ ‖Δx‖² + (1 - θ) Δρ²` over `x = [u; ρ]` is `θ ‖Δu‖² + Δρ²`, valid for every `θ` in (0, 1).
"""
function palc_geometry(geometry::TraceGeometry{E}, width::Integer)::NamedTuple where {E <: Real}
    if geometry.distance == AveragedDistance
        return (; theta = augmented_theta(geometry.state_weight, width), dot = BifurcationKit.DotTheta())
    elseif geometry.distance == SummedDistance
        return (; theta = geometry.state_weight, dot = BifurcationKit.DotTheta(dot))
    end
    weighted = BifurcationKit.PointWeightedDot(FisherWeights(geometry.coordinates), ones(E, width + 1))
    return (; theta = geometry.state_weight, dot = BifurcationKit.DotTheta(weighted))
end
