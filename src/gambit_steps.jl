# EXPERIMENT: Gambit's step control (gambit src/solvers/logit/path.cc, v16.7.0) in Gambit's coordinates; not for landing.
using LinearAlgebra: norm
using LogExpFunctions: log1pexp, logsumexp

"""The log probabilities of a pass's choices at `logits`: each logit is a two-way choice, `log σ(±z)`."""
function log_probabilities(::Any, logits::AbstractVector{E})::AbstractVector{E} where {E <: Real}
    return [.-log1pexp.(.-logits); .-log1pexp.(logits)]
end

"""Each player's log probabilities `[0; logits] .- logsumexp([0; logits])`, players in order."""
function log_probabilities(gains::NormalFormGains, logits::AbstractVector{E})::AbstractVector{E} where {E <: Real}
    return reduce(vcat, [[zero(E); view(logits, range)] .- logsumexp([zero(E); view(logits, range)]) for range in logit_ranges(gains)])
end

"""A branch point `[w; ρ]` in Gambit's coordinates: the log probability of every choice, then `λ = ρ target`."""
function gambit_point(pass::P, point::AbstractVector{E}, target::E)::AbstractVector{E} where {P, E <: Real}
    return [log_probabilities(pass, logits_at(point)); last(point) * target]
end

"""Gambit's first step `hStart` = 0.03 (path.cc) under Gambit's step control, else NaN, which leaves BifurcationKit's ds alone."""
function gambit_first_step(settings::S)::typeof(settings.first_step) where {S}
    if isnothing(settings.step_control) || !settings.step_control.gambit
        return oftype(settings.first_step, NaN)
    end
    return oftype(settings.first_step, 3.0f-2)
end

"""
    gambit_ds!(slot, point, tangent, ds)

The arclength step whose predictor from `point` along `tangent` spans Gambit's next step `h` in Gambit's coordinates.
`h` takes BifurcationKit's step factor since the last call (its distance, contraction and growth rules match path.cc).
"""
function gambit_ds!(slot::S, point::AbstractVector{E}, tangent::AbstractVector{E}, ds::E)::E where {S, E <: Real}
    step = slot.gambit_step * abs(ds) / slot.step_length
    chord = GambitDistance(slot.game.pass!, E(slot.game.target_precision))(point, zero(E), point .+ ds .* tangent, zero(E))
    slot.gambit_step = step
    @debug "gambit h" _group = :gambit h = step predictor_chord = chord ds_in = ds ds_out = abs(ds) * step / chord relative_precision = last(point)
    if !(chord > 0)
        return abs(ds)
    end
    return abs(ds) * step / chord
end

"""The Euclidean distance in Gambit's coordinates between two branch points of `pass` to `target`: PALC's `step_distance`."""
struct GambitDistance{P, E <: Real}
    pass::P
    target::E
end

function (distance::GambitDistance{P, E})(from::AbstractVector{E}, ::Real, to::AbstractVector{E}, ::Real)::E where {P, E <: Real}
    return norm(gambit_point(distance.pass, to, distance.target) .- gambit_point(distance.pass, from, distance.target))
end
