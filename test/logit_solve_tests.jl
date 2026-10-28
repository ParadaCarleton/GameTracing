using ErrorTypes: is_error, unwrap, unwrap_error
using LinearAlgebra: dot

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
        (LOGITNASH_GAMES.matching_pennies, (fill(0.5f0, 2), fill(0.5f0, 2))),
        (ROCK_PAPER_SCISSORS, (uniform, uniform)),
    )
        gains = NormalFormGains(map(table -> Float32.(table), payoffs))
        result = logit_solve(gains)
        @test !is_error(result)
        solved = unwrap(result)
        # Regret sqrt(eps) of the payoff range bounds a dominated strategy's weight near sqrt(eps) of the gap it loses.
        @test largest_gap(solved.profile, equilibrium) < 1.0f-2
        @test GameTracing.max_regret(gains, profile_logits(solved.profile)) <= sqrt(eps(Float32)) * GameTracing.payoff_range(gains)
    end
end

# Games whose branch to maximum_precision 1e6 once stuck at precision 42426 with logits off the map; Gambit ends at
# a mixed equilibrium (λ ≈ 47) and at the pure profile (2, 1, 2) (λ ≈ 1.5).
const STUCK_BRANCH_GAMES = (
    (
        Float32[-0.4899592 0.32297286 -1.141685; -3.5441031 -1.129417 2.330057; 6.2916627 -3.8632107 1.6651082],
        Float32[3.267848 -0.037896026 -1.4952189; -6.062257 1.8497384 2.1450086; 1.3001815 5.7432065 0.7123295],
    ),
    (
        cat(Float32[2.85158 -8.173897; 0.008694828 -0.9346101], Float32[-6.837869 1.001581; 2.4771934 -2.2187471]; dims = 3),
        cat(Float32[3.9944906 -3.9142199; 6.1622057 0.3223508], Float32[5.5092583 1.8642528; 2.5665112 -2.1118705]; dims = 3),
        cat(Float32[-0.05680923 1.9781215; -3.3232822 1.0387676], Float32[-3.1284914 -0.33552215; 2.5984588 -5.7526417]; dims = 3),
    ),
)

"""The precision `λ` that best explains `logits` as `λ G(logits)` of `gains`: the least-squares fit, for a diagnostic."""
function implied_precision(gains::NormalFormGains, logits::AbstractVector{E})::E where {E <: Real}
    width = length(logits)
    gain = zeros(E, width)
    gains(gain, zeros(E, width, width), zeros(E, width), logits, one(E))
    return dot(logits, gain) / dot(gain, gain)
end

"""
The first accepted point of `logit_solve_branch`'s trace of `payoffs` whose mixed strategies differ from Gambit's at its
precision by more than `tolerance`, with the precision before it: where the trace leaves Gambit's branch.
"""
function branch_departure(payoffs::Tuple, tolerance::E)::NamedTuple where {E <: Real}
    gains = NormalFormGains(payoffs)
    settings = GameTracing.gambit_settings(0.03f0, 1.1f0)
    width = GameTracing.logit_count(gains)
    recorder = GameTracing.BranchRecorder(
        gains,
        1.0f6,
        sqrt(eps(E)) * GameTracing.payoff_range(gains),
        fill(E(NaN32), settings.max_steps + 1),
        zeros(E, width, settings.max_steps + 1),
    )
    GameTracing.trace_endpoint(GameTracing.LogitGame(gains, 1.0f6), width, settings; keep_going = recorder)
    recorded = GameTracing.recorded_points(recorder)
    # fatou-ignore call-arity: Base `view` takes any number of indices (jolars/fatou#125)
    logit_points = eachcol(view(recorder.logits, :, recorded))
    points = zip(view(recorder.precisions, recorded), logit_points)
    previous = zero(E)
    for (point, (precision, logits)) in enumerate(points)
        ours = GameTracing.mixed_strategies(gains, logits)
        theirs = gambit_strategies(payoffs, precision).strategies
        if largest_gap(ours, theirs) > tolerance
            return (; point, points = length(recorded), first_precisions = first(view(recorder.precisions, recorded), 12), implied = implied_precision.((gains,), first(collect(logit_points), 12)), precision, previous, gap = largest_gap(ours, theirs), ours, theirs, logits = copy(logits))
        end
        previous = precision
    end
    return (; point = 0, points = length(recorded), first_precisions = first(view(recorder.precisions, recorded), 12), implied = implied_precision.((gains,), first(collect(logit_points), 12)), precision = previous)
end

@testset "logit_solve_branch: each point solves the logit map at its precision, and the last meets maxregret" begin
    rng = fresh_rng()
    random_games = [ntuple(_ -> 3 .* randn(rng, Float32, counts), length(counts)) for counts in ((3, 3), (2, 2, 2))]
    for payoffs in (STUCK_BRANCH_GAMES..., random_games...)
        gains = NormalFormGains(payoffs)
        result = logit_solve_branch(gains)
        @test !is_error(result)
        if is_error(result)
            # Which shorter traces settle tells a failure of the far target from a failure of the branch.
            shorter = [ceiling => logit_solve_branch(gains; maximum_precision = ceiling) for ceiling in (1.0f2, 1.0f4)]
            @error "logit_solve_branch failed" payoffs failure = unwrap_error(result) shorter departure = branch_departure(payoffs, 1.0f-3)
            continue
        end
        branch = unwrap(result)
        @test all(profile -> profile.lam > 0, branch)
        for profile in branch
            logits = profile_logits(profile.profile)
            @test residual_norm(gains, logits, profile.lam) < 1.0f-3 * (1 + maximum(abs, logits))
        end
        @test GameTracing.max_regret(gains, profile_logits(last(branch).profile)) <= sqrt(eps(Float32)) * GameTracing.payoff_range(gains)
        # The last point is on Gambit's branch, logit_solve returns it, and logit_solve_lambda finds it at its precision.
        @test largest_gap(last(branch).profile, gambit_strategies(payoffs, last(branch).lam).strategies) < 1.0f-3
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
