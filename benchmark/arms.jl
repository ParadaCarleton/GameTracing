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

"""
A `Case` whose map is redrawn mid-trace: `case` holds the map before, `after` and `reference_after` the map after in
Float32 and Float64, and the trace swaps at its first accepted point at or past relative precision `swap_level`.
"""
struct SwapCase{C <: Case, A, R, E <: Real}
    case::C
    after::A
    reference_after::R
    swap_level::E
end

"""`swap`'s maps and level as a `SwapCase` named `name`, with `reference_before` and `reference_after` its maps in Float64."""
function swap_case(name::AbstractString, swap::MapSwap, reference_before::Any, reference_after::Any)::SwapCase
    return SwapCase(Case(String(name), swap.before, reference_before, swap.width, swap.precision), swap.after, reference_after, swap.swap_level)
end

"""A random 3-player game whose payoffs move by about 0.05 at relative precision 1/2, as a mid-trace redraw moves a sampled map."""
function swap_case(seed::Integer)::SwapCase
    rng = Xoshiro(1000 + seed)
    before = random_payoffs(rng, 3)
    after = map(table -> table .+ randn(rng, size(table)...) ./ 20, before)
    swap = MapSwap(NormalFormPass(map(table -> Float32.(table), before)), NormalFormPass(map(table -> Float32.(table), after)), 3.0f0, 0.5f0, 3)
    return swap_case(string("swap ", seed), swap, NormalFormPass(before), NormalFormPass(after))
end

"""`case`'s fine Float64 reference from the flat start."""
case_reference(case::Case)::NamedTuple = branch_reference(case.reference_pass, case.width, case.precision)

"""A swap's reference: the map before's branch to the swap level, then the map after's branch on from that point."""
function case_reference(swap::SwapCase)::NamedTuple
    (; case) = swap
    level = case.precision * swap.swap_level
    before = branch_reference(case.reference_pass, case.width, level)
    return branch_reference(swap.reference_after, case.width, case.precision; start = before.logits, start_precision = level)
end

case_name(case::Case)::String = case.name
case_name(swap::SwapCase)::String = swap.case.name

"""`pass` with Float64 parameters, for a reference."""
widened(pass::LinearGainPass)::LinearGainPass = LinearGainPass(pass.interaction .* float(1), pass.stand_alone .* float(1))
widened(pass::ThresholdPass)::ThresholdPass = ThresholdPass(pass.amplitude * float(1), pass.threshold * float(1))

"""The fixed-map games: LogitNash's, the overshot game, the fold game and random games."""
function fixed_cases()::Vector{Case}
    return vcat(
        Case[normal_form_case(string(name), payoffs, 4) for (name, payoffs) in pairs(LOGITNASH_GAMES)],
        Case[normal_form_case("overshot 3", OVERSHOT_GAME, 3), normal_form_case("overshot 3.1", OVERSHOT_GAME, 31 // 10)],
        Case[Case("fold", FOLD_GAME, widened(FOLD_GAME), 1, 2.0f0)],
        Case[random_case(seed) for seed in 1:24],
    )
end

"""The map-swap games: the near and far swaps and random ones."""
function swap_cases()::Vector{SwapCase}
    return vcat(
        SwapCase[
            swap_case(name, swap, widened(swap.before), widened(swap.after))
            for (name, swap) in (("near swap", NEAR_SWAP), ("far swap", FAR_SWAP))
        ],
        SwapCase[swap_case(seed) for seed in 1:6],
    )
end

"""A logger that sends each trace record's message and `reason` (`:none` without one) into `sink`."""
struct TraceCounter <: AbstractLogger
    sink::Channel{Pair{String, Symbol}}
end

min_enabled_level(::TraceCounter)::LogLevel = Debug
shouldlog(::TraceCounter, ::LogLevel, ::Module, group::Any, ::Any)::Bool = group === :trace
catch_exceptions(::TraceCounter)::Bool = false

function handle_message(counter::TraceCounter, ::LogLevel, message::Any, ::Module, ::Any, ::Any, ::Any, ::Any; keywords...)::Nothing
    put!(counter.sink, string(message) => Symbol(get(keywords, :reason, :none)))
    return nothing
end

# Checks GameTracing's corrector callback logs when it stops or refuses a corrector; contraction refusals happen in
# BifurcationKit's CorrectorQuality and show only as halved steps.
const REFUSAL_CHECKS = (:reach, :kl, :angle, :nonfinite)
const REFUSAL_MESSAGES = ("corrector refused", "corrector stopped")

"""`case` traced under `settings`; a `SwapCase` redraws its map through `SwapAt`. A `Function`, as `with_logger` requires."""
struct TraceRun{C, S <: TraceSettings} <: Function
    case::C
    settings::S
end

function (trace_run::TraceRun{<:Case})()::GameTracing.TraceResult{Float32}
    (; case, settings) = trace_run
    return trace_endpoint(LogitGame(case.pass, case.precision), case.width, settings)
end

function (trace_run::TraceRun{<:SwapCase})()::GameTracing.TraceResult{Float32}
    (; settings) = trace_run
    (; case, after, swap_level) = trace_run.case
    return trace_endpoint(LogitGame(case.pass, case.precision), case.width, settings; redraw = SwapAt(LogitGame(after, case.precision), swap_level))
end

"""First-action probabilities of `logits`."""
function first_choices(logits::AbstractVector{E})::Vector{E} where {E <: Real}
    return inv.(one(E) .+ exp.(-logits))
end

"""
What one trace did: first-action probabilities (empty on failure), its status, accepted steps, corrector iterates,
bytes, refusals per check (`REFUSAL_CHECKS`) and the check that refused last (`:none` if none did).
"""
function traced(case::Union{Case, SwapCase}, settings::TraceSettings)::NamedTuple
    trace_run = TraceRun(case, settings)
    trace_run()
    bytes = @allocated trace_run()
    sink = Channel{Pair{String, Symbol}}(typemax(Int))
    result = with_logger(trace_run, TraceCounter(sink))
    close(sink)
    records = collect(sink)
    steps, iterations = count(record -> first(record) == "trace point", records), count(record -> first(record) == "corrector iterate", records)
    refusals = [last(record) for record in records if first(record) in REFUSAL_MESSAGES]
    counts = NamedTuple{REFUSAL_CHECKS}(map(check -> count(==(check), refusals), REFUSAL_CHECKS))
    last_refusal = last([:none; refusals])
    common = (; steps, iterations, bytes, refusals = counts, last_refusal)
    if is_error(result)
        return (; choices = Float32[], status = string(unwrap_error(result).cause), common...)
    end
    return (; choices = first_choices(unwrap(result).logits), status = "reached", common...)
end

"""The largest probability gap between two profiles, NaN when either is empty."""
function profile_gap(left::AbstractVector, right::AbstractVector)::Float32
    if isempty(left) || isempty(right)
        return NaN32
    end
    return Float32(maximum(abs, left .- right))
end

"""Every variant's row on `case`, printed, with a SIGNAL for each pair of reached endpoints over 1e-3 apart."""
function case_rows(case::Union{Case, SwapCase}, variants::AbstractVector)::Vector{NamedTuple}
    reference = case_reference(case)
    # A branch that folds back past the target has no natural-continuation reference: its gap is NaN.
    kind = ifelse(reference.reached, "reference", "folded")
    rows = NamedTuple[
        (; case = case_name(case), label, outcome..., kind, pitchfork = Symbol(case_name(case)) in PITCHFORK_GAMES,
            swapped = case isa SwapCase,
            gap = ifelse(reference.reached, profile_gap(outcome.choices, reference.first_choices), NaN32))
        for (label, outcome) in (label => traced(case, settings) for (label, settings) in variants)
    ]
    for row in rows
        @printf(
            "ROW | %-24s | %-14s | %-13s | steps %5d | iterations %6d | refused reach %d kl %d angle %d nonfinite %d, last %s | gap %.2e (%s) | bytes %d\n",
            row.case, row.label, row.status, row.steps, row.iterations, row.refusals..., row.last_refusal, row.gap, row.kind, row.bytes,
        )
    end
    reached = [row for row in rows if !isempty(row.choices)]
    for (index, left) in pairs(reached), right in reached[(index + 1):end]
        if profile_gap(left.choices, right.choices) > 1.0f-3
            println("SIGNAL | ", case_name(case), " | ", left.label, " vs ", right.label, " | ", @sprintf("%.3e", profile_gap(left.choices, right.choices)))
        end
    end
    flush(stdout)
    return rows
end

"""`reduction` of `values`, NaN when there are none."""
function statistic(reduction::Any, values::AbstractVector)::Float32
    if isempty(values)
        return NaN32
    end
    return Float32(reduction(values))
end

"""
One SUMMARY line for variant `label` over the `rows` of one game group (`group` names it); the gap statistics skip
pitchfork games, whose branch is ambiguous. Failures list the check that refused last before each stall.
"""
function print_summary(group::AbstractString, label::AbstractString, rows::AbstractVector)::Nothing
    mine = [row for row in rows if row.label == label]
    gaps = [row.gap for row in mine if row.status == "reached" && !isnan(row.gap) && !row.pitchfork]
    failures = join(sort([string(row.status, ":", row.last_refusal) for row in mine if row.status != "reached"]), " ")
    @printf(
        "SUMMARY %-5s | %-14s | reached %2d/%2d | off-reference %2d | steps %6d | iterations %7d | gap median %.1e worst %.1e | bytes mean %d | failed %s\n",
        group, label, count(row -> row.status == "reached", mine), length(mine), count(>(1.0f-3), gaps),
        sum(row.steps for row in mine), sum(row.iterations for row in mine), statistic(median, gaps),
        statistic(maximum, gaps), round(Int, sum(row.bytes for row in mine) / length(mine)), failures,
    )
    return nothing
end

"""
Run the benchmark over `scope`: `"full"` every game and variant, `"quick"` one fixed and one swapped game under the
shipped settings and the first arm, the load check CI runs. Throws unless every game and variant gave a row.
"""
function main(output::AbstractString, scope::AbstractString)::Nothing
    variants = ["shipped" => TraceSettings{Float32}(); arm_variants()]
    cases = vcat(fixed_cases(), swap_cases())
    if scope == "quick"
        cases, variants = [first(fixed_cases()), first(swap_cases())], variants[1:2]
    elseif scope != "full"
        throw(ArgumentError("scope must be \"full\" or \"quick\", got $(repr(scope))"))
    end
    rows = reduce(vcat, [case_rows(case, variants) for case in cases])
    if length(rows) != length(cases) * length(variants)
        error("expected $(length(cases) * length(variants)) rows, got $(length(rows))")
    end
    io = open(output, "w")
    println(io, "game,variant,status,steps,iterations,refused_reach,refused_kl,refused_angle,refused_nonfinite,last_refusal,gap,bytes,reference,swapped")
    for row in rows
        println(io, join((row.case, row.label, row.status, row.steps, row.iterations, row.refusals..., row.last_refusal, row.gap, row.bytes, row.kind, row.swapped), ','))
    end
    close(io)
    # Fixed-map and map-swap games are summarized apart: a swap's refusals come from the redraw, not the geometry.
    for (group, swapped) in (("fixed", false), ("swap", true)), (label, _) in variants
        print_summary(group, label, [row for row in rows if row.swapped == swapped])
    end
    return nothing
end

main(get(ARGS, 1, joinpath(@__DIR__, "arms.csv")), get(ARGS, 2, "full"))
