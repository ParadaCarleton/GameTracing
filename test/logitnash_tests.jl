using LogitNash: LogitNash
using Random: RandomDevice, Xoshiro

# LogitNash.jl's games with two actions per player (test/test_e2e.jl and its README), as integer payoff tables.
const LOGITNASH_GAMES = (
    battle_of_the_sexes = ([3 0; 0 2], [2 0; 0 3]),
    coordination_wiki = ([2 1; 1 2], [4 3; 3 4]),
    coordination_pure = ([8 0; 0 8], [8 0; 0 8]),
    coordination_assurance = ([8 0; 0 5], [8 0; 0 5]),
    stag_hunt = ([8 0; 7 5], [8 7; 0 5]),
    prisoners_dilemma = ([5 0; 8 1], [5 8; 0 1]),
    matching_pennies = ([1 -1; -1 1], [-1 1; 1 -1]),
    three_prisoners = (
        cat([3 1; 5 2], [1 0; 2 1]; dims = 3),
        cat([3 5; 1 2], [1 2; 0 1]; dims = 3),
        cat([3 1; 1 0], [5 2; 2 1]; dims = 3),
    ),
)

"""
LogitNash's principal-branch QRE of integer `payoffs` past `precision`: `(; first_choices, precision)`, each player's
probability of its first action and the precision LogitNash stopped at, which may pass the one asked for.
"""
function logitnash_equilibrium(payoffs::Tuple, precision::Real)::NamedTuple
    profile, status = LogitNash.solve(map(float, payoffs); stop_lambda = float(precision), stop_eps = zero(float(precision)))
    return (; first_choices = Float32[first(strategy) for strategy in profile], precision = Float32(status.lambda))
end

"""GameTracing's first-action probabilities for integer `payoffs` at `precision`, or the `TraceFailure`."""
function traced_first_choices(
    payoffs::Tuple,
    precision::E,
    settings::TraceSettings{E},
)::Union{Vector{E}, TraceFailure} where {E <: Real}
    pass = NormalFormPass(map(table -> E.(table), payoffs))
    traced = trace_endpoint(LogitGame(pass, precision), length(payoffs), settings)
    if traced isa TraceFailure
        return traced
    end
    return inv.(1 .+ exp.(-traced.logits))
end

@testset "NormalFormPass: the logit slope is the derivative of the response" begin
    rng = Xoshiro(rand(RandomDevice(), UInt))
    for payoffs in values(LOGITNASH_GAMES)
        pass = NormalFormPass(map(table -> float.(table), payoffs))
        logits = randn(rng, length(payoffs))
        step = inv(float(10^6))
        slope = pass(logits, float(3 // 2)).logit_slope
        for (rival, _) in pairs(logits)
            nudge = step .* (eachindex(logits) .== rival)
            difference = (pass(logits .+ nudge, float(3 // 2)).response .- pass(logits .- nudge, float(3 // 2)).response) ./ (2 * step)
            @test maximum(abs, difference .- slope[:, rival]) < inv(float(10^6))
        end
        @test pass(logits, float(3 // 2)).precision_slope ≈ pass(logits, 1).response
    end
end

@testset "LogitNash's equilibria solve NormalFormPass: both read payoffs the same way" begin
    for payoffs in values(LOGITNASH_GAMES), precision in (1, 4)
        reference = logitnash_equilibrium(payoffs, precision)
        # In probability space: LogitNash's profile may round a choice to certainty, an infinite logit.
        logits = log.(reference.first_choices) .- log1p.(-reference.first_choices)
        response = NormalFormPass(map(table -> Float32.(table), payoffs))(logits, reference.precision).response
        @test maximum(abs, inv.(1 .+ exp.(-response)) .- reference.first_choices) < 1.0f-4
    end
end

# Games symmetric under relabeling actions (battle of the sexes also swaps players): the principal branch meets an exact
# pitchfork (battle of the sexes at precision 0.808), past which the branch is ambiguous; LogitNash keeps the symmetric arm.
const PITCHFORK_GAMES = (:battle_of_the_sexes, :coordination_wiki, :coordination_pure)

@testset "the trace reaches LogitNash's equilibrium on LogitNash's games" begin
    for (name, payoffs) in pairs(LOGITNASH_GAMES), precision in (1, 4)
        if name in PITCHFORK_GAMES
            continue
        end
        reference = logitnash_equilibrium(payoffs, precision)
        for settings in (TraceSettings{Float32}(), TraceSettings{Float32}(tangent = SecantTangent))
            traced = traced_first_choices(payoffs, reference.precision, settings)
            @test traced isa Vector{Float32}
            if traced isa Vector{Float32}
                @test maximum(abs, traced .- reference.first_choices) < 1.0f-3
            else
                @info "trace failed" name precision traced.cause
            end
        end
    end
end

@testset "past an exact pitchfork the trace ends at an equilibrium or fails as Singular" begin
    for name in PITCHFORK_GAMES, settings in (TraceSettings{Float32}(), TraceSettings{Float32}(tangent = SecantTangent))
        pass = NormalFormPass(map(table -> Float32.(table), LOGITNASH_GAMES[name]))
        traced = trace_endpoint(LogitGame(pass, 4.0f0), length(pass.payoffs), settings)
        # Step halving can bisect onto the branch point itself, where the bordered system is exactly singular.
        if traced isa Traced
            @test residual_norm(pass, traced.logits, 4.0f0) < 1.0f-4
        else
            @test traced.cause == GameTracing.Singular
        end
    end
end

"""
The principal-branch first-action probabilities of `pass` at `precision` by natural continuation in steps of `step`, a
few Newton iterations each. It follows an unfolded pitchfork's turn while the turn spans many steps; it fails at a fold.
"""
function natural_continuation(pass::NormalFormPass, precision::Real, step::Real)::Vector
    logits = zeros(eltype(first(pass.payoffs)), length(pass.payoffs))
    for level in range(zero(precision), precision; step), _ in 1:3
        gains = pass(logits, level)
        logits = logits .- (one(gains.logit_slope) .- gains.logit_slope) \ (logits .- gains.response)
    end
    return inv.(1 .+ exp.(-logits))
end

@testset "a pitchfork unfolded by a payoff perturbation: the trace turns with the principal branch" begin
    # LogitNash's steps cross onto the symmetric arm in about 1 % of these games (5 of 400 measured), so the reference
    # is a natural continuation; the perturbation keeps the turn many of its steps wide.
    rng = Xoshiro(rand(RandomDevice(), UInt))
    for _ in 1:10
        scale = (1 + rand(rng)) / 20
        payoffs = map(table -> table .+ scale .* randn(rng, 2, 2), map(float, LOGITNASH_GAMES.battle_of_the_sexes))
        reference = natural_continuation(NormalFormPass(payoffs), 4, inv(float(10^4)))
        traced = traced_first_choices(payoffs, 4.0f0, TraceSettings{Float32}())
        @test traced isa Vector{Float32}
        if traced isa Vector{Float32}
            @test maximum(abs, traced .- reference) < 1.0f-3
        end
    end
end

@testset "random games with two actions each: the trace follows LogitNash's principal branch" begin
    seed = rand(RandomDevice(), UInt)
    rng = Xoshiro(seed)
    for trial in 1:20
        players = rand(rng, 2:3)
        payoffs = ntuple(_ -> 3 .* randn(rng, ntuple(_ -> 2, players)), players)
        reference = logitnash_equilibrium(payoffs, 3)
        traced = traced_first_choices(payoffs, reference.precision, TraceSettings{Float32}())
        @test traced isa Vector{Float32}
        if traced isa Vector{Float32}
            @test maximum(abs, traced .- reference.first_choices) < 1.0f-3
        else
            @info "trace failed" seed trial payoffs traced.cause
        end
    end
end
