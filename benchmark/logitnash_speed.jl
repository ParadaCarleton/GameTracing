# Wall time of GameTracing's `logit_solve` (Float64 and Float32) against LogitNash's `solve` on random normal-form
# games. Each run stops at the first point whose largest regret is at most a tolerance times the game's payoff range.
# Usage: julia --project=benchmark benchmark/logitnash_speed.jl rows.csv [seeds per size] [seconds per timing]
# Prints a row per run as it finishes and rewrites rows.csv after each game, then prints a summary per size.
# Run it under `timeout`: a killed run keeps every finished game in rows.csv.
using GameTracing: NormalFormGains, logit_solve, logit_solve_branch
using ErrorTypes: is_error, unwrap, unwrap_error
using LogitNash: LogitNash
using Pkg: Pkg
using Chairmarks: @be
using CSV: CSV
using LinearAlgebra: dot
using Printf: @printf
using Random: Xoshiro
using Statistics: median

# LogitNash's element type, and the rows' float type.
const WIDE = widen(Float32)
const SIZES = ((2, 2), (3, 3), (5, 5), (10, 10), (20, 20), (2, 2, 2), (3, 3, 3), (5, 5, 5), (2, 2, 2, 2), (3, 3, 3, 3), (2, 2, 2, 2, 2))
# A run whose single timed call takes longer than this is timed once; a faster one is sampled by Chairmarks.
const SLOW_SECONDS = 0.5f0

"""GameTracing's `logit_solve` in element type `E`."""
struct Tracing{E <: Real} end

"""LogitNash's `solve`, which runs in Float64."""
struct Nash end

# Float32 rounds a probability near 1 to about 6e-8, which floors its regret near 1e-6; it runs at 1e-4 only.
const RUNS = (
    (Tracing{WIDE}(), 1.0f-4), (Tracing{Float32}(), 1.0f-4), (Nash(), 1.0f-4),
    (Tracing{WIDE}(), 1.0f-6), (Nash(), 1.0f-6),
)

function solver_name(::Tracing{E})::String where {E}
    return string("GameTracing ", E)
end

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

"""`solver`'s input on `payoffs` at relative `tolerance`: GameTracing's gains and tolerance, or LogitNash's payoffs and absolute `stop_eps`."""
function solver_input(::Tracing{E}, payoffs::Tuple, tolerance::Real)::Tuple where {E <: Real}
    return (NormalFormGains(map(table -> E.(table), payoffs)), tolerance)
end

solver_input(::Nash, payoffs::Tuple, tolerance::Real)::Tuple = (payoffs, tolerance * payoff_range(payoffs))

"""`solver`'s result on `input` (`solver_input`)."""
solve(::Tracing, input::Tuple)::Any = logit_solve(first(input); maxregret = last(input))
solve(::Nash, input::Tuple)::Tuple = LogitNash.solve(first(input); stop_iters = 10^7, stop_lambda = WIDE(1.0f6), stop_eps = last(input))

"""
`(; strategies, outcome, lam, steps)` of `solver`'s `result` on `input`. `strategies` is each player's strategy in
Float64, or `nothing` when the run missed the regret bound. `steps` is GameTracing's recorded branch points, LogitNash's
iterations.
"""
function run_outcome(::Tracing, input::Tuple, result::Any)::NamedTuple
    if is_error(result)
        return (; strategies = nothing, outcome = string(unwrap_error(result).cause), lam = WIDE(NaN32), steps = 0)
    end
    profile = unwrap(result)
    steps = length(unwrap(logit_solve_branch(first(input); maxregret = last(input))))
    return (; strategies = map(strategy -> WIDE.(strategy), profile.profile), outcome = "Ok", lam = WIDE(profile.lam), steps)
end

function run_outcome(::Nash, input::Tuple, result::Tuple)::NamedTuple
    strategies, status = result
    solved = status.regret <= last(input)
    outcome = solved ? "Ok" : status.stall ? "Stall" : "RegretAboveBound"
    return (; strategies = solved ? Tuple(strategies) : nothing, outcome, lam = WIDE(status.lambda), steps = status.iteration)
end

"""The git tree hash of the BifurcationKit this run loads, so each row names the fork commit it measured."""
function bifurcationkit_tree()::String
    return only(info.tree_hash for info in values(Pkg.dependencies()) if info.name == "BifurcationKit")
end

"""`(; seconds, bytes)` of `solver` on `input`: one timed call when it is slow, else the Chairmarks median over `budget` seconds."""
function timed(solver::Any, input::Tuple, budget::Real)::NamedTuple
    single = @timed solve(solver, input)
    if single.time > SLOW_SECONDS
        return (; seconds = single.time, bytes = single.bytes)
    end
    sample = median(@be input Base.Fix1(solve, solver) seconds = budget)
    return (; seconds = sample.time, bytes = sample.bytes)
end

"""The largest probability gap between two profiles, NaN when either is missing."""
profile_gap(left::Tuple, right::Tuple)::WIDE = maximum(maximum.(abs, left .- right))
profile_gap(::Any, ::Any)::WIDE = WIDE(NaN32)

"""The regret of `strategies` on `payoffs` over the payoff range, NaN when the solver failed."""
relative_regret(payoffs::Tuple, strategies::Tuple)::WIDE = profile_regret(payoffs, strategies) / payoff_range(payoffs)
relative_regret(::Tuple, ::Nothing)::WIDE = WIDE(NaN32)

"""
One row per run (`RUNS`) on the game of strategy counts `counts` from `seed`. The first call compiles and gives the
strategies; `timed` measures later calls. The gap is to LogitNash at the same tolerance.
"""
function game_rows(counts::Tuple, seed::Integer, budget::Real, tree::AbstractString)::Vector{NamedTuple}
    payoffs = random_game(counts, seed)
    inputs = [solver_input(solver, payoffs, tolerance) for (solver, tolerance) in RUNS]
    outcomes = [run_outcome(solver, input, solve(solver, input)) for ((solver, _), input) in zip(RUNS, inputs)]
    references = Dict(tolerance => run.strategies for ((solver, tolerance), run) in zip(RUNS, outcomes) if solver isa Nash)
    rows = NamedTuple[]
    for ((solver, tolerance), input, run) in zip(RUNS, inputs, outcomes)
        timing = timed(solver, input, budget)
        row = (
            size = join(counts, "x"), seed = seed, solver = solver_name(solver), tolerance = tolerance,
            solved = !isnothing(run.strategies), outcome = run.outcome, lam = run.lam, steps = run.steps,
            seconds = timing.seconds, bytes = timing.bytes, regret = relative_regret(payoffs, run.strategies),
            gap_to_logitnash = profile_gap(run.strategies, references[tolerance]), bifurcationkit_tree = tree,
        )
        println(join(values(row), '\t'))
        flush(stdout)
        push!(rows, row)
    end
    return rows
end

"""The largest entry of `gaps` that is not NaN, NaN when there is none. A NaN `init` would make every result NaN."""
function largest_gap(gaps::AbstractVector)::WIDE
    known = filter(!isnan, gaps)
    return isempty(known) ? WIDE(NaN32) : maximum(known)
end

"""Print, per game size and run, the games solved, median steps, seconds and bytes, and the largest gap to LogitNash."""
function print_summary(rows::AbstractVector)::Nothing
    @printf("%-10s %-20s %8s %6s %8s %12s %12s %10s\n", "size", "solver", "tol", "solved", "steps", "median s", "median B", "max gap")
    for size in unique(row.size for row in rows), (solver, tolerance) in RUNS
        chosen = [row for row in rows if row.size == size && row.solver == solver_name(solver) && row.tolerance == tolerance]
        @printf(
            "%-10s %-20s %8.0e %3d/%-2d %8.0f %12.3e %12.3e %10.2e\n",
            size, solver_name(solver), tolerance, count(row -> row.solved, chosen), length(chosen),
            median(row.steps for row in chosen), median(row.seconds for row in chosen), median(row.bytes for row in chosen),
            largest_gap([row.gap_to_logitnash for row in chosen]),
        )
    end
    return nothing
end

function main(path::AbstractString, seeds::Integer, budget::Real)::Nothing
    tree = bifurcationkit_tree()
    println(stderr, "BifurcationKit tree ", tree)
    println(
        join(
            (
                "size", "seed", "solver", "tolerance", "solved", "outcome", "lam", "steps", "seconds", "bytes", "regret",
                "gap_to_logitnash", "bifurcationkit_tree",
            ), '\t',
        )
    )
    rows = NamedTuple[]
    for counts in SIZES, seed in 1:seeds
        append!(rows, game_rows(counts, seed, budget, tree))
        CSV.write(path, rows)
    end
    print_summary(rows)
    return nothing
end

main(ARGS[1], parse(Int, get(ARGS, 2, "10")), parse(WIDE, get(ARGS, 3, "0.5")))
