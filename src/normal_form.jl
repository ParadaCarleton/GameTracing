using LinearAlgebra: mul!

"""
    NormalFormGains(payoffs)

The `pass!` of a normal-form game with any number of strategies per player. `payoffs[p][s]` is player `p`'s payoff at
the pure profile `s`; every array has one type and one size, each player's strategy count along its own axis. Player
`p`'s logits are `z_pk = log(π_pk / π_p1)` for its strategies `k ≥ 2`, against its first strategy, stacked player by
player (`logit_ranges`). Its gains are `G_pk = U_p(k, π) - U_p(1, π)`, its expected payoffs against the rivals' mixed
strategies, which do not depend on the precision. A pass writes into work arrays from `pass_workspace`; called
without them, it builds its own.
"""
struct NormalFormGains{N, A <: AbstractArray{<:Real, N}}
    payoffs::NTuple{N, A}

    function NormalFormGains(payoffs::NTuple{N, A})::NormalFormGains{N, A} where {N, A <: AbstractArray{<:Real, N}}
        if !allequal(size.(payoffs))
            throw(ArgumentError("every player's payoff array must have one size; got $(size.(payoffs))"))
        end
        return new{N, A}(payoffs)
    end
end

function NormalFormGains(payoffs::Tuple)::NormalFormGains
    throw(ArgumentError("payoffs must be arrays of one type with one axis per player; got $(typeof.(payoffs))"))
end

"""Each player's range in the stacked logits of `gains`: its strategies after the first."""
function logit_ranges(gains::NormalFormGains{N})::NTuple{N, UnitRange{Int}} where {N}
    counts = size(first(gains.payoffs)) .- 1
    ends = cumsum(counts)
    return UnitRange.(ends .- counts .+ 1, ends)
end

"""The number of logits of `gains`: every player's strategy count less one, summed."""
function logit_count(gains::NormalFormGains)::Integer
    return last(last(logit_ranges(gains)))
end

"""
The work arrays of a `gains` pass at logits of element type `E`: each player's mixed strategy, and room for one player's
expected payoffs, its payoff sensitivities to one rival's strategies, and their means.
"""
function pass_workspace(gains::NormalFormGains, ::AbstractVector{E})::NamedTuple where {E <: Real}
    counts = size(first(gains.payoffs))
    widest = maximum(counts)
    return (;
        strategies = zeros.(E, counts),
        expected = zeros(E, widest),
        sensitivity = zeros(E, widest, widest),
        means = zeros(E, widest),
    )
end

"""Write into `strategy` the mixed strategy `softmax([0; logits])` of a player whose logits are against its first strategy."""
function reference_softmax!(strategy::AbstractVector{E}, logits::AbstractVector{E})::AbstractVector{E} where {E <: Real}
    shift = max(zero(E), maximum(logits; init = zero(E)))
    strategy[begin] = exp(-shift)
    view(strategy, (firstindex(strategy) + 1):lastindex(strategy)) .= exp.(logits .- shift)
    strategy ./= sum(strategy)
    return strategy
end

"""Each player's mixed strategy at the stacked `logits` of `gains`, in fresh vectors."""
function mixed_strategies(gains::NormalFormGains, logits::AbstractVector{E})::Tuple where {E <: Real}
    return reference_softmax!.(zeros.(E, size(first(gains.payoffs))), view.((logits,), logit_ranges(gains)))
end

"""The probability that every player but `player` and `skipped` plays its strategy in `profile`."""
function rival_weight(
    strategies::NTuple{N, AbstractVector{E}},
    profile::CartesianIndex{N},
    player::Integer,
    skipped::Integer,
)::E where {N, E <: Real}
    weight = one(E)
    for (other, strategy) in pairs(strategies)
        if other != player && other != skipped
            weight *= strategy[profile[other]]
        end
    end
    return weight
end

function (gains::NormalFormGains)(
    work::NamedTuple,
    gain::AbstractVector,
    gain_slope::AbstractMatrix,
    gain_precision_slope::AbstractVector,
    logits::AbstractVector{E},
    ::Real,
)::Nothing where {E <: Real}
    ranges = logit_ranges(gains)
    reference_softmax!.(work.strategies, view.((logits,), ranges))
    fill!(gain_slope, zero(E))
    fill!(gain_precision_slope, zero(E))
    for (player, payoff) in pairs(gains.payoffs)
        expected = view(work.expected, axes(payoff, player))
        fill!(expected, zero(E))
        for profile in CartesianIndices(payoff)
            expected[profile[player]] += payoff[profile] * rival_weight(work.strategies, profile, player, 0)
        end
        view(gain, ranges[player]) .= view(expected, (firstindex(expected) + 1):lastindex(expected)) .- first(expected)
        for (rival, rival_range) in pairs(ranges)
            if rival != player
                # fatou-ignore call-arity: Base `view` takes any number of indices (jolars/fatou#125)
                write_gain_slope!(view(gain_slope, ranges[player], rival_range), payoff, work, player, rival)
            end
        end
    end
    return nothing
end

function (gains::NormalFormGains)(
    gain::AbstractVector,
    gain_slope::AbstractMatrix,
    gain_precision_slope::AbstractVector,
    logits::AbstractVector,
    precision::Real,
)::Nothing
    return gains(pass_workspace(gains, logits), gain, gain_slope, gain_precision_slope, logits, precision)
end

"""
    write_gain_slope!(block, payoff, work, player, rival)

`∂G_pk / ∂z_qj` of `player` `p` in the logits of `rival` `q`, into `block`, at the strategies in `work`. With
`D_ki = ∂U_p(k) / ∂π_qi` and `∂π_qi / ∂z_qj = π_qi (δ_ij - π_qj)`, `∂U_p(k) / ∂z_qj = π_qj (D_kj - Σ_i D_ki π_qi)`;
a gain takes its strategy's row less the first strategy's.
"""
function write_gain_slope!(
    block::AbstractMatrix,
    payoff::AbstractArray,
    work::NamedTuple,
    player::Integer,
    rival::Integer,
)::Nothing
    rival_strategy = work.strategies[rival]
    # fatou-ignore call-arity: Base `view` takes any number of indices (jolars/fatou#125)
    sensitivity = view(work.sensitivity, axes(payoff, player), axes(payoff, rival))
    fill!(sensitivity, zero(eltype(sensitivity)))
    for profile in CartesianIndices(payoff)
        sensitivity[profile[player], profile[rival]] += payoff[profile] * rival_weight(work.strategies, profile, player, rival)
    end
    means = mul!(view(work.means, axes(payoff, player)), sensitivity, rival_strategy)
    own, other = axes(sensitivity)
    own_later, later = (first(own) + 1):last(own), (first(other) + 1):last(other)
    # fatou-ignore call-arity: Base `view` takes any number of indices (jolars/fatou#125)
    first_row = view(sensitivity, first(own):first(own), later)
    # fatou-ignore call-arity: Base `view` takes any number of indices (jolars/fatou#125)
    block .= transpose(view(rival_strategy, later)) .* (view(sensitivity, own_later, later) .- view(means, own_later) .- first_row .+ first(means))
    return nothing
end
