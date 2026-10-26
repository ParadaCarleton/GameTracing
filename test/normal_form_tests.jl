using ErrorTypes: is_error, unwrap, unwrap_error
using LogitNash: LogitNash

"""LogitNash's mixed strategies of `payoffs` past `precision`, and the precision it stopped at, which may pass it."""
function logitnash_profile(payoffs::Tuple, precision::Real)::NamedTuple
    profile, status = LogitNash.solve(map(float, payoffs); stop_lambda = float(precision), stop_eps = zero(float(precision)))
    return (; strategies = map(strategy -> Float32.(strategy), profile), precision = Float32(status.lambda))
end

"""The largest difference in any probability between two profiles of mixed strategies."""
function profile_gap(left::Tuple, right::Tuple)::eltype(first(left))
    return maximum(maximum.(abs, left .- right))
end

"""Each player's log odds against its first strategy, stacked: the logits of a profile of mixed strategies."""
function stacked_log_odds(strategies::Tuple)::AbstractVector
    return reduce(vcat, map(strategy -> log.(view(strategy, 2:lastindex(strategy)) ./ first(strategy)), strategies))
end

"""
    strategies_cross_checked!(payoffs, precision)

Trace the normal-form game `payoffs` to LogitNash's stopping precision past `precision` and test the endpoint against
`branch_reference`, with LogitNash as a cross-check (`cross_checked!`). A branch that folds back in precision has no
reference; every endpoint must also solve the map to the corrector's tolerance (`settled_endpoint`).
"""
function strategies_cross_checked!(payoffs::Tuple, precision::Real)::Nothing
    cross = logitnash_profile(payoffs, precision)
    gains = NormalFormGains(map(table -> Float32.(table), payoffs))
    settings = TraceSettings{Float32}()
    result = trace_endpoint(LogitGame(gains, cross.precision), logit_count(gains), settings)
    @test !is_error(result)
    if is_error(result)
        @info "trace failed" payoffs unwrap_error(result).cause
        return nothing
    end
    traced = mixed_strategies(gains, unwrap(result).logits)
    reference_gains = NormalFormGains(map(float, payoffs))
    reference = branch_reference(reference_gains, logit_count(reference_gains), cross.precision)
    if reference.reached
        @test profile_gap(traced, mixed_strategies(reference_gains, reference.logits)) < 1.0f-3
    end
    @test settled_endpoint(gains, unwrap(result).logits, cross.precision, settings)
    if profile_gap(traced, cross.strategies) >= 1.0f-3 && reference.reached
        @test profile_gap(mixed_strategies(reference_gains, reference.logits), cross.strategies) >= 1.0f-3
    end
    return nothing
end

@testset "NormalFormGains with two strategies each is NormalFormPass with its logits negated" begin
    rng = fresh_rng()
    for payoffs in values(LOGITNASH_GAMES)
        floats = map(table -> float.(table), payoffs)
        logits = randn(rng, length(payoffs))
        level = float(3 // 2)
        against_first = evaluated(NormalFormGains(floats), -logits, level)
        against_second = evaluated(NormalFormPass(floats), logits, level)
        @test against_first.response ≈ -against_second.response
        @test against_first.logit_slope ≈ against_second.logit_slope
    end
end

"""Player `player`'s expected payoff of each of its strategies, summed over every pure profile."""
function summed_expected_payoffs(payoff::AbstractArray, strategies::Tuple, player::Integer)::AbstractVector
    rivals = filter(!=(player), eachindex(strategies))
    return [
        sum(
            payoff[profile] * prod(strategies[rival][profile[rival]] for rival in rivals)
            for profile in CartesianIndices(payoff) if profile[player] == strategy
        )
        for strategy in axes(payoff, player)
    ]
end

"""The bytes one call of `pass` allocates at `logits` and `precision`, after a first call compiles it."""
function pass_bytes(pass::Any, logits::AbstractVector{E}, precision::Real)::Integer where {E <: Real}
    width = length(logits)
    gain, gain_slope, gain_precision_slope = zeros(E, width), zeros(E, width, width), zeros(E, width)
    pass(gain, gain_slope, gain_precision_slope, logits, precision)
    return @allocated pass(gain, gain_slope, gain_precision_slope, logits, precision)
end

@testset "NormalFormGains: each player's gains, stacked in player order, are its expected payoffs less its first strategy's" begin
    rng = fresh_rng()
    for counts in ((3, 3, 3), (2, 3, 4), (4, 2, 3), (3, 2))
        payoffs = ntuple(_ -> 3 .* randn(rng, counts), length(counts))
        gains = NormalFormGains(payoffs)
        logits = randn(rng, logit_count(gains))
        strategies = mixed_strategies(gains, logits)
        expected = summed_expected_payoffs.(payoffs, (strategies,), eachindex(payoffs))
        stacked = reduce(vcat, [view(payoff, 2:lastindex(payoff)) .- first(payoff) for payoff in expected])
        @test evaluated(gains, logits, 1).response ≈ stacked
    end
end

@testset "a NormalFormGains pass allocates nothing" begin
    rng = fresh_rng()
    gains = NormalFormGains(ntuple(_ -> randn(rng, Float32, 3, 2, 4), 3))
    @test pass_bytes(gains, randn(rng, Float32, logit_count(gains)), 2.0f0) == 0
end

@testset "NormalFormGains: the gain slope is the derivative of the gains, at any strategy counts" begin
    rng = fresh_rng()
    for counts in ((3, 3), (2, 3, 4), (4, 2), (1, 3), (3, 2, 2))
        payoffs = ntuple(_ -> 3 .* randn(rng, counts), length(counts))
        gains = NormalFormGains(payoffs)
        logits = randn(rng, logit_count(gains))
        step = inv(float(10^6))
        level = float(3 // 2)
        slope = evaluated(gains, logits, level).logit_slope
        for (rival, _) in pairs(logits)
            nudge = step .* (eachindex(logits) .== rival)
            upper, lower = evaluated(gains, logits .+ nudge, level), evaluated(gains, logits .- nudge, level)
            @test maximum(abs, (upper.response .- lower.response) ./ (2 * step) .- slope[:, rival]) < inv(float(10^6))
        end
        @test evaluated(gains, logits, level).precision_slope ≈ evaluated(gains, logits, 1).response
    end
end

@testset "mixed_strategies are probability vectors whose log odds against the first strategy are the logits" begin
    rng = fresh_rng()
    gains = NormalFormGains(ntuple(_ -> randn(rng, 3, 4, 2), 3))
    @test logit_count(gains) == 2 + 3 + 1
    logits = 5 .* randn(rng, logit_count(gains))
    strategies = mixed_strategies(gains, logits)
    @test length.(strategies) == (3, 4, 2)
    @test all(strategy -> sum(strategy) ≈ 1 && all(>=(0), strategy), strategies)
    @test stacked_log_odds(strategies) ≈ logits
    # A player with one strategy has no logits.
    @test logit_count(NormalFormGains((randn(rng, 1, 3), randn(rng, 1, 3)))) == 2
end

@testset "NormalFormGains refuses payoff arrays of different sizes" begin
    @test_throws ArgumentError NormalFormGains((zeros(2, 3), zeros(3, 2)))
    @test_throws ArgumentError NormalFormGains((zeros(2, 3), zeros(Int, 2, 3)))
    @test_throws ArgumentError NormalFormGains((zeros(2, 3), zeros(2, 3), zeros(2, 3)))
end

@testset "rock-paper-scissors traces to the uniform mixture at every precision" begin
    beats = [0 -1 1; 1 0 -1; -1 1 0]
    game = LogitGame(NormalFormGains((Float32.(beats), Float32.(-beats))), 5.0f0)
    traced = unwrap(trace_endpoint(game, logit_count(game.pass!), TraceSettings{Float32}()))
    @test profile_gap(mixed_strategies(game.pass!, traced.logits), (fill(1.0f0 / 3, 3), fill(1.0f0 / 3, 3))) < 1.0f-4
end

@testset "games with more than two strategies: the trace follows the principal branch" begin
    rng = fresh_rng()
    for counts in ((3, 3), (2, 3), (3, 2, 2), (4, 3)), _ in 1:4
        strategies_cross_checked!(ntuple(_ -> 3 .* randn(rng, counts), length(counts)), 3)
    end
end
