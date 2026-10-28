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

"""The Euclidean distance in Gambit's coordinates between two branch points of `pass` to `target`: PALC's `step_distance`."""
struct GambitDistance{P, E <: Real}
    pass::P
    target::E
end

function (distance::GambitDistance{P, E})(from::AbstractVector{E}, ::Real, to::AbstractVector{E}, ::Real)::E where {P, E <: Real}
    return norm(gambit_point(distance.pass, to, distance.target) .- gambit_point(distance.pass, from, distance.target))
end
