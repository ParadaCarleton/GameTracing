using ErrorTypes: is_error, unwrap, unwrap_error

"""
Gambit's logit QRE of the normal-form game `payoffs` at `precision`, from `pygambit.qre.logit_solve_lambda`:
`(; precision, strategies)`, each player's mixed strategy in `precision`'s type.
"""
function gambit_strategies(payoffs::Tuple, precision::E)::NamedTuple where {E <: Real}
    gambit, numpy = pyimport("pygambit"), pyimport("numpy")
    game = gambit.Game.from_arrays((numpy.asarray(float.(table)) for table in payoffs)...)
    profile = first(gambit.qre.logit_solve_lambda(game; lam = [float(precision)]))
    strategies = Tuple([pyconvert(E, profile.profile[strategy]) for strategy in player.strategies] for player in game.players)
    return (; precision = pyconvert(E, profile.lam), strategies)
end

"""The largest difference in any probability between two profiles of mixed strategies."""
function largest_gap(left::Tuple, right::Tuple)::eltype(first(left))
    return maximum(maximum.(abs, left .- right))
end

"""Each player's log odds against its first strategy, stacked: the logits of a profile of mixed strategies."""
function profile_logits(profile::Tuple)::AbstractVector
    return reduce(vcat, [log.(view(strategy, 2:lastindex(strategy)) ./ first(strategy)) for strategy in profile])
end

const ROCK_PAPER_SCISSORS = ([0 -1 1; 1 0 -1; -1 1 0], [0 1 -1; -1 0 1; 1 -1 0])

@testset "logit_solve_lambda agrees with Gambit's logit_solve_lambda at each precision" begin
    rng = fresh_rng()
    random_games = [ntuple(_ -> 3 .* randn(rng, counts), length(counts)) for counts in ((3, 3), (2, 3, 2), (4, 2))]
    games = [LOGITNASH_GAMES.prisoners_dilemma, LOGITNASH_GAMES.matching_pennies, ROCK_PAPER_SCISSORS, random_games...]
    for payoffs in games
        gains = NormalFormGains(map(table -> Float32.(table), payoffs))
        profiles = unwrap(logit_solve_lambda(gains, [1, 4]))
        @test [profile.lam for profile in profiles] == Float32[1, 4]
        for profile in profiles
            oracle = gambit_strategies(payoffs, profile.lam)
            @test largest_gap(profile.profile, oracle.strategies) < 1.0f-3
        end
    end
    @test length(unwrap(logit_solve_lambda(NormalFormGains(map(table -> Float32.(table), ROCK_PAPER_SCISSORS)), 2))) == 1
end

@testset "logit_solve ends at the Nash equilibrium of games whose equilibrium is known" begin
    uniform = fill(1.0f0 / 3, 3)
    for (payoffs, equilibrium) in (
        (LOGITNASH_GAMES.prisoners_dilemma, (Float32[0, 1], Float32[0, 1])),
        (LOGITNASH_GAMES.matching_pennies, (Float32[0.5, 0.5], Float32[0.5, 0.5])),
        (ROCK_PAPER_SCISSORS, (uniform, uniform)),
    )
        gains = NormalFormGains(map(table -> Float32.(table), payoffs))
        result = logit_solve(gains)
        @test !is_error(result)
        solved = unwrap(result)
        # Regret 1e-4 of the payoff range bounds a dominated strategy's weight near 1e-4 of the gap it loses.
        @test largest_gap(solved.profile, equilibrium) < 1.0f-2
        @test GameTracing.max_regret(gains, profile_logits(solved.profile)) <= 1.0f-4 * GameTracing.payoff_range(gains)
    end
end

@testset "logit_solve_branch: each point solves the logit map at its precision, and the last meets maxregret" begin
    rng = fresh_rng()
    for counts in ((3, 3), (2, 2, 2))
        gains = NormalFormGains(ntuple(_ -> 3 .* randn(rng, Float32, counts), length(counts)))
        branch = unwrap(logit_solve_branch(gains))
        @test all(profile -> profile.lam > 0, branch)
        for profile in branch
            logits = profile_logits(profile.profile)
            @test residual_norm(gains, logits, profile.lam) < 1.0f-3 * (1 + maximum(abs, logits))
        end
        @test GameTracing.max_regret(gains, profile_logits(last(branch).profile)) <= 1.0f-4 * GameTracing.payoff_range(gains)
        # logit_solve is the branch's last point, and logit_solve_lambda finds the same point at its precision.
        solved = unwrap(logit_solve(gains))
        @test solved.lam == last(branch).lam
        again = only(unwrap(logit_solve_lambda(gains, solved.lam)))
        @test largest_gap(again.profile, solved.profile) < 1.0f-3
    end
end

@testset "logit_solve_branch fails NoConvergence when maximum_precision comes before maxregret" begin
    gains = NormalFormGains(map(table -> Float32.(table), LOGITNASH_GAMES.prisoners_dilemma))
    failure = unwrap_error(logit_solve_branch(gains; maximum_precision = 1))
    @test failure.cause == GameTracing.NoConvergence
    @test failure.context.reason == :maximum_precision
end

@testset "the solvers refuse Gambit's invalid arguments" begin
    gains = NormalFormGains(map(table -> Float32.(table), LOGITNASH_GAMES.matching_pennies))
    @test_throws ArgumentError logit_solve(gains; maxregret = 0)
    @test_throws ArgumentError logit_solve(gains; first_step = 0)
    @test_throws ArgumentError logit_solve_lambda(gains, 1; max_accel = 0.5f0)
end
