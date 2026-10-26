# The trace-geometry arms (test/arms.jl) against the shipped trace on GameTracing's games, each beside a
# fine-step Float64 natural-continuation reference (`branch_reference`). Prints a ROW per game and variant,
# a SIGNAL per pair of reached endpoints more than 1e-3 apart, and a SUMMARY per variant; writes rows to ARGS[1].
using GameTracing
using ErrorTypes: is_error, unwrap, unwrap_error
using Logging: AbstractLogger, Debug, LogLevel, with_logger
import Logging: catch_exceptions, handle_message, min_enabled_level, shouldlog
using Printf: @printf, @sprintf
using Random: Xoshiro
using Statistics: median

const TEST_DIRECTORY = joinpath(@__DIR__, "..", "test")
include(joinpath(TEST_DIRECTORY, "games.jl"))
include(joinpath(TEST_DIRECTORY, "fixtures.jl"))
include(joinpath(TEST_DIRECTORY, "arms.jl"))

"""One benchmark game: its name, the pass in Float32 and in Float64 for the reference, its width and target precision."""
struct Case{P, R, E <: Real}
    name::String
    pass::P
    reference_pass::R
    width::Int
    precision::E
end

"""A two-action normal-form `Case` from integer, rational or float payoff tables."""
function normal_form_case(name::AbstractString, payoffs::Tuple, precision::Real)::Case
    return Case(String(name), NormalFormPass(map(table -> Float32.(table), payoffs)), NormalFormPass(map(float, payoffs)), length(payoffs), Float32(precision))
end

"""`players` random payoff tables of scale 3 from `rng`."""
random_payoffs(rng::Xoshiro, players::Integer)::Tuple = ntuple(_ -> 3 .* randn(rng, ntuple(_ -> 2, players)), players)

"""A random game of 2 to 5 players from seed `seed`."""
function random_case(seed::Integer)::Case
    rng = Xoshiro(seed)
    return normal_form_case(string("random ", seed), random_payoffs(rng, rand(rng, 2:5)), 3)
end

"""A random 3-player game whose payoffs move by about 0.05 at precision 1.5, as a mid-trace redraw moves a sampled map."""
function swap_case(seed::Integer)::Case
    rng = Xoshiro(1000 + seed)
    before = random_payoffs(rng, 3)
    after = map(table -> table .+ randn(rng, size(table)...) ./ 20, before)
    pass = SwappedPass(NormalFormPass(map(table -> Float32.(table), before)), NormalFormPass(map(table -> Float32.(table), after)), 1.5f0)
    return Case(string("swap ", seed), pass, SwappedPass(NormalFormPass(before), NormalFormPass(after), float(3 // 2)), 3, 3.0f0)
end

"""The benchmark's games: LogitNash's, the overshot game, the fold game, random games and games whose map swaps mid-trace."""
function benchmark_cases()::Vector{Case}
    return vcat(
        Case[normal_form_case(string(name), payoffs, 4) for (name, payoffs) in pairs(LOGITNASH_GAMES)],
        Case[normal_form_case("overshot 3", OVERSHOT_GAME, 3), normal_form_case("overshot 3.1", OVERSHOT_GAME, 31 // 10)],
        Case[Case("fold", FOLD_GAME, ThresholdPass(float(6), float(3)), 1, 2.0f0)],
        Case[random_case(seed) for seed in 1:24],
        Case[swap_case(seed) for seed in 1:6],
    )
end

"""A logger that counts the trace's accepted points and corrector iterates into `sink`."""
struct TraceCounter <: AbstractLogger
    sink::Channel{String}
end

min_enabled_level(::TraceCounter)::LogLevel = Debug
shouldlog(::TraceCounter, ::LogLevel, ::Module, group::Any, ::Any)::Bool = group === :trace
catch_exceptions(::TraceCounter)::Bool = false

function handle_message(counter::TraceCounter, ::LogLevel, message::Any, ::Module, ::Any, ::Any, ::Any, ::Any; _...)::Nothing
    put!(counter.sink, string(message))
    return nothing
end

"""`case` traced under `settings`."""
struct TraceRun{C <: Case, S <: TraceSettings}
    case::C
    settings::S
end

(trace_run::TraceRun)()::GameTracing.TraceResult{Float32} = trace_endpoint(LogitGame(trace_run.case.pass, trace_run.case.precision), trace_run.case.width, trace_run.settings)

"""First-action probabilities of `logits`."""
first_choices(logits::AbstractVector{E})::Vector{E} where {E <: Real} = inv.(one(E) .+ exp.(-logits))

"""What one trace did: first-action probabilities (empty on failure), its status, accepted steps, corrector iterates and bytes."""
function traced(case::Case, settings::TraceSettings)::NamedTuple
    trace_run = TraceRun(case, settings)
    trace_run()
    bytes = @allocated trace_run()
    sink = Channel{String}(Inf)
    result = with_logger(trace_run, TraceCounter(sink))
    close(sink)
    messages = collect(sink)
    steps, iterations = count(==("trace point"), messages), count(==("corrector iterate"), messages)
    if is_error(result)
        return (; choices = Float32[], status = string(unwrap_error(result).cause), steps, iterations, bytes)
    end
    return (; choices = first_choices(unwrap(result).logits), status = "reached", steps, iterations, bytes)
end

"""The largest probability gap between two profiles, NaN when either is empty."""
function profile_gap(left::AbstractVector, right::AbstractVector)::Float64
    if isempty(left) || isempty(right)
        return NaN
    end
    return Float64(maximum(abs, left .- right))
end

"""Every variant's row on `case`, printed, with a SIGNAL for each pair of reached endpoints over 1e-3 apart."""
function case_rows(case::Case, variants::AbstractVector)::Vector{NamedTuple}
    reference = branch_reference(case.reference_pass, case.width, case.precision)
    # A branch that folds back past the target has no natural-continuation reference: its gap is NaN.
    kind = ifelse(reference.reached, "reference", "folded")
    rows = NamedTuple[
        (; case = case.name, label, outcome..., kind, pitchfork = Symbol(case.name) in PITCHFORK_GAMES,
            gap = ifelse(reference.reached, profile_gap(outcome.choices, reference.first_choices), NaN))
        for (label, outcome) in (label => traced(case, settings) for (label, settings) in variants)
    ]
    for row in rows
        @printf("ROW | %-24s | %-14s | %-13s | steps %5d | iterations %6d | gap %.2e (%s) | bytes %d\n", row.case, row.label, row.status, row.steps, row.iterations, row.gap, row.kind, row.bytes)
    end
    reached = [row for row in rows if !isempty(row.choices)]
    for (index, left) in pairs(reached), right in reached[(index + 1):end]
        if profile_gap(left.choices, right.choices) > 1.0e-3
            println("SIGNAL | ", case.name, " | ", left.label, " vs ", right.label, " | ", @sprintf("%.3e", profile_gap(left.choices, right.choices)))
        end
    end
    flush(stdout)
    return rows
end

"""`reduction` of `values`, NaN when there are none."""
function statistic(reduction::Any, values::AbstractVector)::Float64
    if isempty(values)
        return NaN
    end
    return Float64(reduction(values))
end

"""One SUMMARY line for variant `label` over `rows`; the gap statistics skip pitchfork games, whose branch is ambiguous."""
function print_summary(label::AbstractString, rows::AbstractVector)::Nothing
    mine = [row for row in rows if row.label == label]
    gaps = [row.gap for row in mine if row.status == "reached" && !isnan(row.gap) && !row.pitchfork]
    failures = join(sort(unique(row.status for row in mine if row.status != "reached")), "/")
    @printf(
        "SUMMARY | %-14s | reached %2d/%2d %-16s | off-reference %2d | steps %6d | iterations %7d | gap median %.1e worst %.1e | bytes mean %d\n",
        label, count(row -> row.status == "reached", mine), length(mine), failures, count(>(1.0e-3), gaps),
        sum(row.steps for row in mine), sum(row.iterations for row in mine), statistic(median, gaps),
        statistic(maximum, gaps), round(Int, sum(row.bytes for row in mine) / length(mine)),
    )
    return nothing
end

function main(output::AbstractString)::Nothing
    variants = ["shipped" => TraceSettings{Float32}(); arm_variants()]
    rows = reduce(vcat, [case_rows(case, variants) for case in benchmark_cases()])
    io = open(output, "w")
    println(io, "game,variant,status,steps,iterations,gap,bytes,reference")
    for row in rows
        println(io, join((row.case, row.label, row.status, row.steps, row.iterations, row.gap, row.bytes, row.kind), ','))
    end
    close(io)
    for (label, _) in variants
        print_summary(label, rows)
    end
    return nothing
end

main(get(ARGS, 1, joinpath(@__DIR__, "arms.csv")))
