using BifurcationKit: BifurcationKit
using LinearAlgebra: dot
using LogExpFunctions: log1pexp, logistic
"""
The trace's step length in a state `x = [w; ρ]` of `k` payoff gaps `w`, at `state_weight` (the gaps' weight):

- `AveragedDistance`: `state_weight ‖Δw‖² / k + (1 - state_weight) Δρ²`.
- `SummedDistance`: `state_weight ‖Δw‖² + Δρ²`.
- `FisherDistance`: `state_weight Σ σ(zᵢ)σ(-zᵢ) Δwᵢ² + Δρ²`, with the Bernoulli Fisher weights of the logits `z = ρ w`.
- `FisherLogPrecisionDistance`: `state_weight Σ σ(zᵢ)σ(-zᵢ) Δwᵢ² + (Δλ / (1 + λ))²` at the precision `λ = ρ P`, `P` the
  target: Fisher's gaps beside the step in `log(1 + λ)` to first order, LogitNash's path variable.

Both Fisher distances freeze their weights at each accepted point, and their reach and angle checks measure in the
summed norm.
"""
@enum StepDistance AveragedDistance SummedDistance FisherDistance FisherLogPrecisionDistance

"""
    TraceGeometry{E}(; distance, state_weight)

The norm a trace steps in: its `StepDistance`, and `state_weight`, the gaps' weight against the relative precision's.
The default is the Fisher distance at `state_weight = 0.01`: of the arms benchmarked, it took the fewest steps,
corrector iterates and bytes, with every endpoint on the reference (`benchmark/results/arms-f3244ba-summary.txt`).
"""
Base.@kwdef struct TraceGeometry{E <: Real}
    distance::StepDistance = FisherDistance
    state_weight::E = 0.01f0
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
    return coordinate_weight(geometry, length(left) - 1) * dot(payoff_gaps(left), payoff_gaps(right)) +
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
    payoff_gaps(weights) .= logistic.(logits) .* logistic.(.-logits)
    weights[end] = one(E)
    return weights
end

"""`(Δλ / (1 + λ))²` per `Δρ²`, `P² / (1 + ρ P)²`, at relative precision `relative_precision` of a trace to precision `target_precision`."""
function log_precision_scale(relative_precision::E, target_precision::E)::E where {E <: Real}
    return (target_precision / (one(E) + relative_precision * target_precision))^2
end

"""
The length in `geometry`'s distance of a unit step in the relative precision at `relative_precision`, on a trace to
`target_precision` (`StepDistance`): `P / (1 + ρ P)` under `FisherLogPrecisionDistance`, else `√level_weight`.
"""
function precision_length(geometry::TraceGeometry{E}, relative_precision::E, target_precision::E)::E where {E <: Real}
    if geometry.distance == FisherLogPrecisionDistance
        return sqrt(log_precision_scale(relative_precision, target_precision))
    end
    return sqrt(level_weight(geometry))
end

"""
PALC's state weight for a `FisherLogPrecisionDistance` trace to `target_precision`: one minus half of `P² / (1 + P)²`,
the smallest `log_precision_scale` over ρ ∈ [0, 1]. PALC's own `Δρ²` term then stays under the distance's, so the
dot's precision weight (`FisherLogPrecisionWeights`) stays positive.
"""
function log_precision_palc_state_weight(target_precision::E)::E where {E <: Real}
    return one(E) - log_precision_scale(one(E), target_precision) / 2
end

"""
`weights!` of a `FisherLogPrecisionDistance` trace's `PointWeightedDot`, under PALC state weight `palc_state_weight`
(`log_precision_palc_state_weight`), both weights at the accepted point:

- each gap: its Fisher weight `σ(z)σ(-z)` times `gap_scale = state_weight / palc_state_weight`;
- the relative precision: `(log_precision_scale - (1 - palc_state_weight)) / palc_state_weight`.

PALC's `palc_state_weight ⟨Δx, Δx⟩ + (1 - palc_state_weight) Δρ²` is then the distance's norm.
"""
struct FisherLogPrecisionWeights{E <: Real}
    gap_scale::E
    palc_state_weight::E
    target_precision::E
end

"""The weight of `Δρ` in `weighting`'s dot at relative precision `relative_precision`."""
function precision_dot_weight(weighting::FisherLogPrecisionWeights{E}, relative_precision::E)::E where {E <: Real}
    scale = log_precision_scale(relative_precision, weighting.target_precision)
    return (scale - (one(E) - weighting.palc_state_weight)) / weighting.palc_state_weight
end

function (weighting::FisherLogPrecisionWeights{E})(weights::AbstractVector{E}, point::BifurcationKit.BorderedArray)::AbstractVector{E} where {E <: Real}
    logits = logits_at(point.u)
    payoff_gaps(weights) .= weighting.gap_scale .* logistic.(logits) .* logistic.(.-logits)
    weights[end] = precision_dot_weight(weighting, last(point.u))
    return weights
end

"""`weights!` of an `AveragedDistance` trace's `PointWeightedDot`: its weights stay those it was built with."""
struct FixedWeights end

function (::FixedWeights)(weights::AbstractVector{E}, ::BifurcationKit.BorderedArray)::AbstractVector{E} where {E <: Real}
    return weights
end

"""
`(; palc_state_weight, dot)`: PALC's `θ` and `dotθ` for `geometry` over a trace of `width` gaps to precision
`target_precision`. PALC's norm of `[Δx; Δρ]`, `x = [w; ρ]`, is `palc_state_weight ⟨Δx, Δx⟩ + (1 - palc_state_weight) Δρ²`.
`palc_state_weight` is `geometry.state_weight`, and the inner product weighs:

- averaged: each gap by `1/k` and `ρ` by zero;
- summed: every coordinate by one;
- Fisher: each gap by its Fisher weight and `ρ` by one.

A `FisherLogPrecisionDistance` trace runs PALC at `log_precision_palc_state_weight` instead (`FisherLogPrecisionWeights`).
Every `state_weight` in (0, 1) is valid, at every width; another throws an `ArgumentError`.
"""
function palc_geometry(geometry::TraceGeometry{E}, width::Integer, target_precision::Real)::NamedTuple where {E <: Real}
    (; state_weight) = geometry
    if !(zero(E) < state_weight < one(E))
        throw(ArgumentError("the state weight $state_weight lies outside (0, 1), where PALC weights the gaps and the precision"))
    end
    if geometry.distance == AveragedDistance
        weights = [fill(inv(E(width)), width); zero(E)]
        return (; palc_state_weight = state_weight, dot = BifurcationKit.DotTheta(BifurcationKit.PointWeightedDot(FixedWeights(), weights)))
    elseif geometry.distance == SummedDistance
        return (; palc_state_weight = state_weight, dot = BifurcationKit.DotTheta(dot))
    elseif geometry.distance == FisherLogPrecisionDistance
        palc_state_weight = log_precision_palc_state_weight(E(target_precision))
        weighting = FisherLogPrecisionWeights(state_weight / palc_state_weight, palc_state_weight, E(target_precision))
        # The flat start's logits are zero, where each Fisher weight is 1/4.
        weights = [fill(weighting.gap_scale / 4, width); precision_dot_weight(weighting, zero(E))]
        return (; palc_state_weight, dot = BifurcationKit.DotTheta(BifurcationKit.PointWeightedDot(weighting, weights)))
    end
    weighted = BifurcationKit.PointWeightedDot(FisherWeights(), ones(E, width + 1))
    return (; palc_state_weight = state_weight, dot = BifurcationKit.DotTheta(weighted))
end

"""PALC's product `state_weight ⟨x, y⟩ + (1 - state_weight) ρₓ ρ_y` of two branch vectors `[x; ρ]` under the trace's `dot`."""
function palc_product(
    dot::BifurcationKit.DotTheta,
    state_weight::E,
    left::AbstractVector{E},
    right::AbstractVector{E},
)::E where {E <: Real}
    return dot(left, right, last(left), last(right), state_weight)
end

"""`vector` under the in-place operator `A` of a `DotTheta`, `⟨x, y⟩ = xᵀ A y`; `nothing` is the identity."""
function applied_operator(::Nothing, vector::AbstractVector{E})::AbstractVector{E} where {E <: Real}
    return vector
end

function applied_operator(apply::Any, vector::AbstractVector{E})::AbstractVector{E} where {E <: Real}
    return apply(vector)
end

"""The row `b` with `b ⋅ τ = palc_product(dot, state_weight, vector, τ)` for every branch vector `τ`."""
function palc_border(dot::BifurcationKit.DotTheta, state_weight::E, vector::AbstractVector{E})::AbstractVector{E} where {E <: Real}
    border = state_weight .* applied_operator(BifurcationKit._get_apply_dot(dot), copy(vector))
    border[end] += (one(E) - state_weight) * last(vector)
    return border
end
