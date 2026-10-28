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
    GambitRescale(slot)

PALC's `step_rescale`: the signed ds whose predictor from `point` along `tangent` spans Gambit's next step `h` in
Gambit's coordinates. `h` takes BifurcationKit's step factor since the ds last used (`slot.step_length`): its distance,
contraction and growth rules match path.cc, and a refused step's halving halves `h`.
"""
struct GambitRescale{S}
    slot::S
end

function (rescale::GambitRescale{S})(point::AbstractVector{E}, tangent::AbstractVector{E}, ds::T)::T where {S, E <: Real, T <: Real}
    slot = rescale.slot
    step = slot.gambit_step * abs(E(ds)) / slot.step_length
    chord = GambitDistance(slot.game.pass!, E(slot.game.target_precision))(point, zero(E), point .+ E(ds) .* tangent, zero(E))
    rescaled = abs(E(ds))
    if chord > 0
        rescaled = rescaled * step / chord
    end
    @debug "gambit h" _group = :gambit h = step predictor_chord = chord ds_in = ds ds_out = rescaled relative_precision = last(point)
    slot.gambit_step = step
    slot.step_length = rescaled
    return copysign(T(rescaled), ds)
end

"""PALC's `step_rescale` for a trace whose `slot` runs Gambit's step control, else `nothing`."""
function gambit_rescale(slot::S)::Union{Nothing, GambitRescale{S}} where {S}
    if !isfinite(slot.gambit_step)
        return nothing
    end
    return GambitRescale(slot)
end

"""The Euclidean distance in Gambit's coordinates between two branch points of `pass` to `target`: PALC's `step_distance`."""
struct GambitDistance{P, E <: Real}
    pass::P
    target::E
end

function (distance::GambitDistance{P, E})(from::AbstractVector{E}, ::Real, to::AbstractVector{E}, ::Real)::E where {P, E <: Real}
    return norm(gambit_point(distance.pass, to, distance.target) .- gambit_point(distance.pass, from, distance.target))
end
