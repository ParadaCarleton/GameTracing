using ErrorTypes: is_error, unwrap, unwrap_error
using Random: AbstractRNG, Xoshiro

"""The largest difference in any probability between two profiles of mixed strategies."""
function largest_gap(left::Tuple, right::Tuple)::eltype(first(left))
    return maximum(maximum.(abs, left .- right))
end

# Gambit's corrector stops at c_tol = 1e-4 in log probabilities (src/solvers/nash/logit/path.cc), so ≤ 1e-4 in probability.
const GAMBIT_PROBABILITY_TOLERANCE = 1.0f-4

"""
The largest probability error of the trace's `profile`: softmax slope 1/2 times its logit error, the corrector's
`logit_tolerance` (a settled corrector's last Newton correction) plus 32 ulps of the largest logit.
"""
function profile_bound(profile::LogitQREProfile{E})::E where {E <: Real}
    return (CorrectorSettings{E}().logit_tolerance + 32 * eps(E) * maximum(abs, profile.logits)) / 2
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
            @test largest_gap(profile.profile, oracle.strategies) < profile_bound(profile) + GAMBIT_PROBABILITY_TOLERANCE
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
        @test GameTracing.max_regret(gains, solved.logits) <= sqrt(eps(Float32)) * GameTracing.payoff_range(gains)
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
            @test settled_logits(gains, profile.logits, profile.lam, TraceSettings{eltype(profile.logits)}())
            @test all(isfinite, profile.logits)
        end
        @test GameTracing.max_regret(gains, last(branch).logits) <= sqrt(eps(Float32)) * GameTracing.payoff_range(gains)
        # The last point is on Gambit's branch, logit_solve returns it, and logit_solve_lambda finds it at its precision.
        @test largest_gap(last(branch).profile, gambit_strategies(payoffs, last(branch).lam).strategies) <
            profile_bound(last(branch)) + GAMBIT_PROBABILITY_TOLERANCE
        solved = unwrap(logit_solve(gains))
        @test solved.lam == last(branch).lam
        again = only(unwrap(logit_solve_lambda(gains, solved.lam)))
        @test largest_gap(again.profile, solved.profile) < profile_bound(again) + profile_bound(solved)
    end
end

@testset "logit_solve_branch fails NoConvergence when maximum_precision comes before maxregret" begin
    gains = NormalFormGains(map(table -> Float32.(table), LOGITNASH_GAMES.prisoners_dilemma))
    failure = unwrap_error(logit_solve_branch(gains; maximum_precision = 1))
    @test failure.cause == GameTracing.NoConvergence
    @test failure.context.reason == :maximum_precision
end

@testset "the solvers return the trace's failure when it runs out of steps" begin
    gains = NormalFormGains(map(table -> Float32.(table), LOGITNASH_GAMES.prisoners_dilemma))
    settings = TraceSettings{Float32}(; max_steps = 1)
    @test unwrap_error(logit_solve_lambda(gains, 100; settings)).cause == GameTracing.NoConvergence
    for failure in unwrap_error.((logit_solve_branch(gains; settings), logit_solve(gains; settings)))
        @test failure.cause == GameTracing.NoConvergence
        # The trace's own failure, not the one logit_solve_branch builds when the trace ends unsettled.
        @test !haskey(failure.context, :maximum_precision)
    end
end

@testset "the solvers refuse Gambit's invalid arguments" begin
    gains = NormalFormGains(map(table -> Float32.(table), LOGITNASH_GAMES.matching_pennies))
    @test_throws ArgumentError logit_solve(gains; maxregret = 0)
    @test_throws ArgumentError logit_solve(gains; first_step = 0)
    @test_throws ArgumentError logit_solve_lambda(gains, 1; max_accel = 0.5f0)
end

@testset "the solvers return profiles in the payoffs' element type and refuse settings in another" begin
    for element in (Float32, float(Int))
        gains = NormalFormGains(map(table -> element.(table), LOGITNASH_GAMES.prisoners_dilemma))
        @test eltype(gains) == element
        @test unwrap(logit_solve(gains)).lam isa element
        @test eltype(unwrap(logit_solve(gains)).logits) == element
        @test eltype(only(unwrap(logit_solve_lambda(gains, 2))).logits) == element
        other = element == Float32 ? float(Int) : Float32
        settings = TraceSettings{other}()
        @test_throws ArgumentError logit_solve_lambda(gains, 2; settings)
        @test_throws ArgumentError logit_solve_branch(gains; settings)
        @test_throws ArgumentError logit_solve(gains; settings)
    end
    # Integer payoffs trace in `float(Int)`, as Julia promotes them.
    integer = NormalFormGains(ROCK_PAPER_SCISSORS)
    @test eltype(integer) == Int
    @test eltype(unwrap(logit_solve(integer)).logits) == float(Int)
    @test eltype(only(unwrap(logit_solve_lambda(integer, 2))).logits) == float(Int)
    @test_throws ArgumentError logit_solve(integer; settings = TraceSettings{Float32}())
end

@testset "LogitQREProfile computes its profile from its logits and takes no profile" begin
    gains = NormalFormGains(map(table -> Float32.(table), LOGITNASH_GAMES.matching_pennies))
    profile = LogitQREProfile(gains, 1.0f0, Float32[0, 0])
    @test profile.profile == GameTracing.mixed_strategies(gains, profile.logits)
    @test_throws MethodError LogitQREProfile(1.0f0, profile.profile, Float32[3, -3])
end
