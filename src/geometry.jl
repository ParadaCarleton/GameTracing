using BifurcationKit: BifurcationKit
using LinearAlgebra: dot
using LogExpFunctions: log1pexp, logistic
"""
The trace's step length in a state `x = [w; ρ]` of `k` payoff gaps `w` at state weight `θ`: `AveragedDistance` is
`θ ‖Δw‖² / k + (1 - θ) Δρ²`, `SummedDistance` `θ ‖Δw‖² + Δρ²`, and `FisherDistance` `θ Σ σ(zᵢ)σ(-zᵢ) Δwᵢ² + Δρ²`
with the Bernoulli Fisher weights of the logits `z = ρ w` frozen at each accepted point. The reach and angle checks
of a `FisherDistance` trace measure in the summed norm.
"""
@enum StepDistance AveragedDistance SummedDistance FisherDistance

"""
    TraceGeometry{E}(; distance, state_weight)

The norm a trace steps in: its `StepDistance`, and `state_weight`, the gaps' weight `θ` against the relative
precision's. The default is PALC's averaged norm at `θ = 1/2`.
"""
Base.@kwdef struct TraceGeometry{E <: Real}
    distance::StepDistance = AveragedDistance
    state_weight::E = 0.5f0
end

"""The weight of each `Δwᵢ²` in `geometry`'s plain norm over `width` gaps (`StepDistance`)."""
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

"""The plain inner product of two displacements `[Δw; Δρ]` in `geometry` (`coordinate_weight`, `level_weight`)."""
function trace_dot(
    left::AbstractVector{E},
    right::AbstractVector{E},
    geometry::TraceGeometry{E},
)::E where {E <: Real}
    return coordinate_weight(geometry, length(left) - 1) * dot(state_of(left), state_of(right)) +
        level_weight(geometry) * last(left) * last(right)
end

"""The distance from `from` to `to`, both `[w; ρ]`, in `geometry`'s plain norm (`trace_dot`)."""
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

"""The largest `bernoulli_kl` over the logits of two points `[w; ρ]` (`logits_at`)."""
function largest_kl(from::AbstractVector{E}, to::AbstractVector{E})::E where {E <: Real}
    return maximum(bernoulli_kl(start, finish) for (start, finish) in zip(logits_at(from), logits_at(to)))
end

"""
`weights!` of a `FisherDistance` trace's `PointWeightedDot`: each gap's Bernoulli Fisher weight `σ(z)σ(-z)` at the
accepted point's logits `z`, and one for the relative precision.
"""
struct FisherWeights end

function (::FisherWeights)(weights::AbstractVector{E}, point::BifurcationKit.BorderedArray)::AbstractVector{E} where {E <: Real}
    logits = logits_at(point.u)
    state_of(weights) .= logistic.(logits) .* logistic.(.-logits)
    weights[end] = one(E)
    return weights
end

"""`weights!` of an `AveragedDistance` trace's `PointWeightedDot`: its weights stay those it was built with."""
struct FixedWeights end

function (::FixedWeights)(weights::AbstractVector{E}, ::BifurcationKit.BorderedArray)::AbstractVector{E} where {E <: Real}
    return weights
end

"""
PALC's `θ` and `dotθ` for `geometry` over a trace of `width` gaps. PALC's norm of `[Δx; Δρ]`, `x = [w; ρ]`, is
`θ ⟨Δx, Δx⟩ + (1 - θ) Δρ²` at the state weight `θ`: the inner product weighs each gap by `1/k` and `ρ` by zero
(averaged), every coordinate by one (summed), or each gap by its Fisher weight and `ρ` by one (Fisher). Every `θ` in
(0, 1) is valid, at every width; another throws an `ArgumentError`.
"""
function palc_geometry(geometry::TraceGeometry{E}, width::Integer)::NamedTuple where {E <: Real}
    theta = geometry.state_weight
    if !(zero(E) < theta < one(E))
        throw(ArgumentError("the state weight $theta lies outside (0, 1), where PALC weights the gaps and the precision"))
    end
    if geometry.distance == AveragedDistance
        weights = [fill(inv(E(width)), width); zero(E)]
        return (; theta, dot = BifurcationKit.DotTheta(BifurcationKit.PointWeightedDot(FixedWeights(), weights)))
    elseif geometry.distance == SummedDistance
        return (; theta, dot = BifurcationKit.DotTheta(dot))
    end
    weighted = BifurcationKit.PointWeightedDot(FisherWeights(), ones(E, width + 1))
    return (; theta, dot = BifurcationKit.DotTheta(weighted))
end
