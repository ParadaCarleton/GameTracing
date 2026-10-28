using ErrorTypes: is_error, unwrap, unwrap_error
using Random: AbstractRNG, Xoshiro

"""The largest difference in any probability between two profiles of mixed strategies."""
function largest_gap(left::Tuple, right::Tuple)::eltype(first(left))
    return maximum(maximum.(abs, left .- right))
end

"""Each player's log odds against its first strategy, stacked: the logits of a profile of mixed strategies."""
function profile_logits(profile::Tuple)::AbstractVector
    return reduce(vcat, [log.(view(strategy, 2:lastindex(strategy)) ./ first(strategy)) for strategy in profile])
end

# Gambit's corrector stops at c_tol = 1e-4 in log probabilities (src/solvers/nash/logit/path.cc), so ≤ 1e-4 in probability.
const GAMBIT_PROBABILITY_TOLERANCE = 1.0f-4

"""The logit residual under which the trace accepts `logits`: the corrector tolerance at ρ = 1, where gap and logit residuals agree."""
function accepted_residual(logits::AbstractVector{E})::E where {E <: Real}
    corrector = CorrectorSettings{E}()
    return GameTracing.corrector_tolerance(corrector.residual_tolerance, corrector.tolerance_floor_ulps, [logits; one(E)])
end

"""The largest probability error of the trace's `profile` of `gains` (`probability_bound`, softmax slope 1/2)."""
function profile_bound(gains::NormalFormGains, profile::LogitQREProfile{E})::E where {E <: Real}
    logits = profile_logits(profile.profile)
    return probability_bound(gains, logits, profile.lam, accepted_residual(logits), 0.5f0)
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
            @test largest_gap(profile.profile, oracle.strategies) < profile_bound(gains, profile) + GAMBIT_PROBABILITY_TOLERANCE
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

# The seed whose branch games once stuck at precision 42426 with logits off the map: Gambit ends the (3, 3) game at a
# mixed equilibrium (λ ≈ 47) and the (2, 2, 2) game at the pure profile (2, 1, 2) (λ ≈ 1.5).
const STUCK_BRANCH_SEED = 0x126798e4d8bed9ec

"""The random games of the branch testset, one per strategy count in `((3, 3), (2, 2, 2))`, drawn from `rng`."""
function branch_games(rng::AbstractRNG)::AbstractVector
    return [ntuple(_ -> 3 .* randn(rng, Float32, counts), length(counts)) for counts in ((3, 3), (2, 2, 2))]
end

@testset "logit_solve_branch: each point solves the logit map at its precision, and the last meets maxregret" begin
    stuck_games = branch_games(Xoshiro(STUCK_BRANCH_SEED))
    # The seed's first payoff, as the failing run logged it.
    @test first(first(first(stuck_games))) == -0.4899592f0
    for payoffs in (stuck_games..., branch_games(fresh_rng())...)
        gains = NormalFormGains(payoffs)
        result = logit_solve_branch(gains)
        @test !is_error(result)
        if is_error(result)
            @error "logit_solve_branch failed" payoffs = repr(payoffs) payoff_bits = map(table -> reinterpret.(UInt32, table), payoffs) failure = unwrap_error(result)
            continue
        end
        branch = unwrap(result)
        @test all(profile -> profile.lam > 0, branch)
        for profile in branch
            logits = profile_logits(profile.profile)
            @test residual_norm(gains, logits, profile.lam) < 2 * accepted_residual(logits)
        end
        @test GameTracing.max_regret(gains, profile_logits(last(branch).profile)) <= sqrt(eps(Float32)) * GameTracing.payoff_range(gains)
        # The last point is on Gambit's branch, logit_solve returns it, and logit_solve_lambda finds it at its precision.
        @test largest_gap(last(branch).profile, gambit_strategies(payoffs, last(branch).lam).strategies) <
            profile_bound(gains, last(branch)) + GAMBIT_PROBABILITY_TOLERANCE
        solved = unwrap(logit_solve(gains))
        @test solved.lam == last(branch).lam
        again = only(unwrap(logit_solve_lambda(gains, solved.lam)))
        @test largest_gap(again.profile, solved.profile) < profile_bound(gains, again) + profile_bound(gains, solved)
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
