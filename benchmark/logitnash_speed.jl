# Wall time of GameTracing's `logit_solve` (Float64 and Float32) against LogitNash's `solve` on random normal-form
# games. Each stops at the first point whose largest regret is at most REGRET times the game's payoff range.
# Usage: julia --project=benchmark benchmark/logitnash_speed.jl rows.csv [seconds per timing]
using GameTracing: NormalFormGains, logit_solve
using ErrorTypes: is_error, unwrap
using LogitNash: LogitNash
using Chairmarks: @be
using CSV: CSV
using LinearAlgebra: dot
using Printf: @printf
using Random: Xoshiro
using Statistics: median
using StructArrays: StructArray

const REGRET = 1.0e-6
const SEEDS = 1:10
const SIZES = ((2, 2), (3, 3), (5, 5), (10, 10), (20, 20), (2, 2, 2), (3, 3, 3), (5, 5, 5), (2, 2, 2, 2), (3, 3, 3, 3), (2, 2, 2, 2, 2))

"""GameTracing's `logit_solve` in element type `E`."""
struct Tracing{E <: Real} end

"""LogitNash's `solve`, which runs in Float64."""
struct Nash end

const SOLVERS = (Tracing{Float64}(), Tracing{Float32}(), Nash())

solver_name(::Tracing{E})::String where {E} = string("GameTracing ", E)
solver_name(::Nash)::String = "LogitNash"

"""A random game with strategy counts `counts`: standard normal payoffs from `Xoshiro(seed)`."""
function random_game(counts::Tuple, seed::Integer)::Tuple
    rng = Xoshiro(seed)
    return ntuple(_ -> randn(rng, counts), length(counts))
end

"""The largest payoff of `payoffs` less its smallest."""
function payoff_range(payoffs::NTuple{N, AbstractArray{E, N}})::E where {N, E <: Real}
    return maximum(maximum, payoffs) - minimum(minimum, payoffs)
end

"""The largest gain any player of `payoffs` makes by moving from `strategies` to one of its pure strategies."""
function profile_regret(payoffs::NTuple{N, AbstractArray{E, N}}, strategies::Tuple)::E where {N, E <: Real}
    worst = zero(E)
    for (player, payoff) in pairs(payoffs)
        expected = zeros(E, size(payoff, player))
        for profile in CartesianIndices(payoff)
            weight = one(E)
            for (other, strategy) in pairs(strategies)
                if other != player
                    weight *= strategy[profile[other]]
                end
            end
            expected[profile[player]] += payoff[profile] * weight
        end
        worst = max(worst, maximum(expected) - dot(strategies[player], expected))
    end
    return worst
end

"""The game `solver` is timed on: GameTracing's gains in its element type, or LogitNash's payoffs and absolute `stop_eps`."""
function solver_input(::Tracing{E}, payoffs::Tuple)::NormalFormGains where {E <: Real}
    return NormalFormGains(map(table -> E.(table), payoffs))
end

solver_input(::Nash, payoffs::Tuple)::Tuple = (payoffs, REGRET * payoff_range(payoffs))

"""GameTracing's approximate equilibrium of `gains` at regret REGRET, as a `Result`."""
solve_tracing(gains::NormalFormGains)::Any = logit_solve(gains; maxregret = REGRET)

"""LogitNash's approximate equilibrium of `task = (payoffs, stop_eps)`, with its step and precision caps out of reach."""
function solve_logitnash(task::Tuple)::Tuple
    return LogitNash.solve(first(task); stop_iters = 10^7, stop_lambda = 1.0e6, stop_eps = last(task))
end

"""Each player's strategy from `solver` on `input` in Float64, or `nothing` when it missed the regret bound."""
function solved_strategies(::Tracing, input::NormalFormGains)::Union{Nothing, Tuple}
    result = solve_tracing(input)
    if is_error(result)
        return nothing
    end
    return map(strategy -> Float64.(strategy), unwrap(result).profile)
end

function solved_strategies(::Nash, input::Tuple)::Union{Nothing, Tuple}
    strategies, status = solve_logitnash(input)
    if status.regret > last(input)
        return nothing
    end
    return Tuple(strategies)
end

"""The median Chairmarks sample of `solver` on `input`, timed for about `seconds`."""
function timed(::Tracing, input::NormalFormGains, seconds::Real)::Any
    return median(@be input solve_tracing seconds = seconds)
end

function timed(::Nash, input::Tuple, seconds::Real)::Any
    return median(@be input solve_logitnash seconds = seconds)
end

"""The largest probability gap between two profiles, NaN when either is missing."""
profile_gap(left::Tuple, right::Tuple)::Float64 = maximum(maximum.(abs, left .- right))
profile_gap(::Any, ::Any)::Float64 = NaN

"""The regret of `strategies` on `payoffs` over the payoff range, NaN when the solver failed."""
relative_regret(payoffs::Tuple, strategies::Tuple)::Float64 = profile_regret(payoffs, strategies) / payoff_range(payoffs)
relative_regret(::Tuple, ::Nothing)::Float64 = NaN

"""One row per solver for the game of strategy counts `counts` from `seed`."""
function game_rows(counts::Tuple, seed::Integer, seconds::Real)::StructArray
    payoffs = random_game(counts, seed)
    inputs = solver_input.(SOLVERS, (payoffs,))
    profiles = solved_strategies.(SOLVERS, inputs)
    samples = timed.(SOLVERS, inputs, seconds)
    reference = last(profiles)
    return StructArray(
        (
            size = join(counts, "x"),
            seed = seed,
            solver = solver_name(solver),
            solved = !isnothing(profile),
            seconds = sample.time,
            bytes = sample.bytes,
            regret = relative_regret(payoffs, profile),
            gap_to_logitnash = profile_gap(profile, reference),
        ) for (solver, profile, sample) in zip(SOLVERS, profiles, samples)
    )
end

"""The largest finite entry of `gaps`, NaN when there is none."""
largest_gap(gaps::AbstractVector)::Float64 = maximum(filter(!isnan, gaps); init = NaN)

"""Print, per game size and solver, the games solved, median seconds, median bytes and the largest gap to LogitNash."""
function print_summary(rows::StructArray)::Nothing
    @printf("%-10s %-20s %6s %12s %12s %10s\n", "size", "solver", "solved", "median s", "median B", "max gap")
    for size in unique(rows.size), solver in solver_name.(SOLVERS)
        chosen = rows[(rows.size .== size) .& (rows.solver .== solver)]
        @printf(
            "%-10s %-20s %3d/%-2d %12.3e %12.3e %10.2e\n",
            size, solver, count(chosen.solved), length(chosen), median(chosen.seconds), median(chosen.bytes),
            largest_gap(chosen.gap_to_logitnash),
        )
    end
    return nothing
end

function main(path::AbstractString, seconds::Real)::Nothing
    rows = reduce(vcat, [game_rows(counts, seed, seconds) for counts in SIZES for seed in SEEDS])
    CSV.write(path, rows)
    print_summary(rows)
    return nothing
end

main(ARGS[1], parse(Float64, get(ARGS, 2, "1")))
