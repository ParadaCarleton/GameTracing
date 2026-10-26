"""
    NormalFormGains(payoffs)

The `pass!` of a normal-form game with any number of strategies per player. `payoffs[p][s]` is player `p`'s payoff at
the pure profile `s`; every array has one size, each player's strategy count along its own axis. Player `p`'s logits
are `z_pk = log(π_pk / π_p1)` for its strategies `k ≥ 2`, against its first strategy, stacked player by player
(`logit_ranges`). Its gains are `G_pk = U_p(k, π) - U_p(1, π)`, its expected payoffs against the rivals' mixed
strategies, which do not depend on the precision.
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

"""The mixed strategy `softmax([0; logits])` of a player whose logits are against its first strategy."""
function reference_softmax(logits::AbstractVector{E})::AbstractVector{E} where {E <: Real}
    shift = max(zero(E), maximum(logits; init = zero(E)))
    weights = [exp(-shift); exp.(logits .- shift)]
    return weights ./ sum(weights)
end

"""Each player's mixed strategy at the stacked `logits` of `gains`."""
function mixed_strategies(gains::NormalFormGains, logits::AbstractVector{E})::Tuple where {E <: Real}
    return reference_softmax.(view.((logits,), logit_ranges(gains)))
end

"""The probability that every player but `player` and `skipped` plays its strategy in `profile`."""
function rival_weight(strategies::Tuple, profile::CartesianIndex, player::Integer, skipped::Integer)::eltype(first(strategies))
    weight = one(eltype(first(strategies)))
    for (other, strategy) in pairs(strategies)
        if other != player && other != skipped
            weight *= strategy[profile[other]]
        end
    end
    return weight
end

function (gains::NormalFormGains)(
    gain::AbstractVector,
    gain_slope::AbstractMatrix,
    gain_precision_slope::AbstractVector,
    logits::AbstractVector{E},
    ::Real,
)::Nothing where {E <: Real}
    # ponytail: allocates each player's expected payoffs and sensitivities per call; preallocate if a profile shows it.
    strategies = mixed_strategies(gains, logits)
    ranges = logit_ranges(gains)
    fill!(gain_slope, zero(E))
    fill!(gain_precision_slope, zero(E))
    for (player, payoff) in pairs(gains.payoffs)
        expected = zeros(E, size(payoff, player))
        for profile in CartesianIndices(payoff)
            expected[profile[player]] += payoff[profile] * rival_weight(strategies, profile, player, 0)
        end
        view(gain, ranges[player]) .= view(expected, 2:lastindex(expected)) .- first(expected)
        for (rival, rival_range) in pairs(ranges)
            if rival != player
                # fatou-ignore call-arity: Base `view` takes any number of indices (jolars/fatou#125)
                write_gain_slope!(view(gain_slope, ranges[player], rival_range), payoff, strategies, player, rival)
            end
        end
    end
    return nothing
end

"""
    write_gain_slope!(block, payoff, strategies, player, rival)

`∂G_pk / ∂z_qj` of `player` `p` in the logits of `rival` `q`, into `block`. With `D_ki = ∂U_p(k) / ∂π_qi` and
`∂π_qi / ∂z_qj = π_qi (δ_ij - π_qj)`, `∂U_p(k) / ∂z_qj = π_qj (D_kj - Σ_i D_ki π_qi)`; a gain takes its strategy's
row less the first strategy's.
"""
function write_gain_slope!(
    block::AbstractMatrix,
    payoff::AbstractArray,
    strategies::Tuple,
    player::Integer,
    rival::Integer,
)::Nothing
    rival_strategy = strategies[rival]
    sensitivity = zeros(eltype(rival_strategy), size(payoff, player), size(payoff, rival))
    for profile in CartesianIndices(payoff)
        sensitivity[profile[player], profile[rival]] += payoff[profile] * rival_weight(strategies, profile, player, rival)
    end
    slopes = (sensitivity .- sensitivity * rival_strategy) .* transpose(rival_strategy)
    own, other = axes(slopes)
    later = (first(other) + 1):last(other)
    # fatou-ignore call-arity: Base `view` takes any number of indices (jolars/fatou#125)
    block .= view(slopes, (first(own) + 1):last(own), later) .- view(slopes, first(own):first(own), later)
    return nothing
end

"""
    normal_form_game(payoffs, target_precision)

The `LogitGame` of the normal-form game `payoffs` (`NormalFormGains`), traced to `target_precision`. Its trace has
`logit_count(game.pass!)` logits, and `mixed_strategies(game.pass!, logits)` reads each player's mixed strategy.
"""
function normal_form_game(payoffs::Tuple, target_precision::Real)::LogitGame
    return LogitGame(NormalFormGains(payoffs), target_precision)
end
