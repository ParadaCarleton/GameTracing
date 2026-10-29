# The trace-geometry arms (test/arms.jl) and the shipped trace on GameTracing's games, each beside a fine-step Float64
# reference (`branch_reference`), and LogitNash 0.3.0 on the normal-form games. Prints a ROW per game and variant, a
# SIGNAL per pair of endpoints over 1e-3 apart and a SUMMARY per variant; writes rows to ARGS[1].
using GameTracing
using ErrorTypes: Result, is_error, unwrap, unwrap_error
using LogitNash: LogitNash
using Logging: AbstractLogger, Debug, LogLevel, with_logger
import Logging: catch_exceptions, handle_message, min_enabled_level, shouldlog
using Printf: @sprintf
using Random: AbstractRNG, Xoshiro
using Statistics: median

const TEST_DIRECTORY = joinpath(@__DIR__, "..", "test")
include(joinpath(TEST_DIRECTORY, "games.jl"))
include(joinpath(TEST_DIRECTORY, "fixtures.jl"))
include(joinpath(TEST_DIRECTORY, "arms.jl"))

"""A benchmark game: a fixed map (`Case`) or a map redrawn mid-trace (`SwapCase`)."""
abstract type BenchmarkCase end

"""One fixed-map game: its name, the pass in Float32 and in Float64 for the reference, its width and target precision."""
struct Case{P, R, E <: Real} <: BenchmarkCase
    name::String
    pass::P
    reference_pass::R
    width::Int
    precision::E
end

"""A two-action normal-form `Case` from integer, rational or float payoff tables."""
function normal_form_case(name::AbstractString, payoffs::Tuple, precision::Real)::BenchmarkCase
    return Case(String(name), NormalFormPass(map(single_precision, payoffs)), NormalFormPass(map(float, payoffs)), length(payoffs), Float32(precision))
end

"""LogitNash's game `name` with `payoffs`, to precision 4."""
logitnash_case(name::Symbol, payoffs::Tuple)::BenchmarkCase = normal_form_case(string(name), payoffs, 4)

"""A random game of 2 to 5 players from seed `seed`."""
function random_case(seed::Integer)::BenchmarkCase
    rng = Xoshiro(seed)
    return normal_form_case(string("random ", seed), random_payoffs(rng, rand(rng, 2:5)), 3)
end

"""
A game whose map is redrawn mid-trace: `case` holds the map before, `after` and `reference_after` the map after in
Float32 and Float64, and the trace swaps at its first accepted point at or past relative precision `swap_level`.
"""
struct SwapCase{C <: Case, A, R, E <: Real} <: BenchmarkCase
    case::C
    after::A
    reference_after::R
    swap_level::E
end

"""`swap`'s maps and level as a `SwapCase` named `name`, with `reference_before` and `reference_after` its maps in Float64."""
function swap_case(name::AbstractString, swap::MapSwap, reference_before::Any, reference_after::Any)::BenchmarkCase
    return SwapCase(Case(String(name), swap.before, reference_before, swap.width, swap.precision), swap.after, reference_after, swap.swap_level)
end

"""The random swap `seed` (`random_swap`) as a `SwapCase`."""
swap_case(seed::Integer)::BenchmarkCase = drawn_swap_case(string("swap ", seed), random_swap(seed))

"""`drawn`, a `random_swap`, as a `SwapCase` named `name`."""
function drawn_swap_case(name::AbstractString, drawn::T)::BenchmarkCase where {T <: NamedTuple}
    return swap_case(name, drawn.swap, drawn.reference_before, drawn.reference_after)
end

"""`case`'s fine Float64 reference from the flat start."""
case_reference(case::Case)::NamedTuple = branch_reference(case.reference_pass, case.width, case.precision)

"""
The relative precision at which `swap`'s trace under `settings` redraws: its first accepted point at or past the swap
level, from the same trace stopped there. A trace that fails before the swap level gives its last accepted point.
"""
function swap_precision(swap::SwapCase{C, A, R, E}, settings::S)::E where {C, A, R, E, S <: TraceSettings}
    (; case) = swap
    stopped = trace_endpoint(LogitGame(case.pass, case.precision), case.width, settings; keep_going = StopAt(swap.swap_level))
    return E(unwrap_error(stopped).context.relative_precision)
end

"""
A swap's reference under `settings`: the map before's branch to the precision where that trace redraws
(`swap_precision`), then the map after's branch on from that point. A redraw happens at an accepted point, so each
variant's reference swaps where its own trace does.
"""
function case_reference(swap::W, settings::S)::NamedTuple where {W <: SwapCase, S <: TraceSettings}
    (; case) = swap
    level = case.precision * swap_precision(swap, settings)
    before = branch_reference(case.reference_pass, case.width, level)
    return branch_reference(swap.reference_after, case.width, case.precision; start = before.logits, start_precision = level)
end

"""The reference every variant of `case` shares: a fixed map's own, none for a swap, whose reference is per variant."""
shared_reference(case::Case)::NamedTuple = case_reference(case)
shared_reference(::SwapCase)::Nothing = nothing

"""The reference of `case`'s trace under `settings`: the `shared` one, or for a swap its own (`case_reference`)."""
variant_reference(::Case, ::TraceSettings, shared::NamedTuple)::NamedTuple = shared
variant_reference(swap::SwapCase, settings::TraceSettings, ::Nothing)::NamedTuple = case_reference(swap, settings)

"""The gap kind of `reference`: a branch that folds back past the target has no natural-continuation reference."""
reference_kind(reference::NamedTuple)::AbstractString = ifelse(reference.reached, "reference", "folded")

"""`pass` with its parameters widened (Float32 to Float64), for a reference."""
(widened(pass::LinearGainPass{E})::LinearGainPass{widen(E)}) where {E} = LinearGainPass(widen.(pass.interaction), widen.(pass.stand_alone))
(widened(pass::ThresholdPass{E})::ThresholdPass{widen(E)}) where {E} = ThresholdPass(widen(pass.amplitude), widen(pass.threshold))

case_name(case::Case)::AbstractString = case.name
case_name(swap::SwapCase)::AbstractString = swap.case.name

"""The fixed-map games: LogitNash's, the overshot game, the fold game and random games."""
function fixed_cases()::AbstractVector{BenchmarkCase}
    return BenchmarkCase[
        map(logitnash_case, keys(LOGITNASH_GAMES), values(LOGITNASH_GAMES))...,
        normal_form_case("overshot 3", OVERSHOT_GAME, 3),
        normal_form_case("overshot 3.1", OVERSHOT_GAME, 31 // 10),
        Case("fold", FOLD_GAME, widened(FOLD_GAME), 1, 2.0f0),
        map(random_case, 1:24)...,
    ]
end

"""The map-swap games: the near and far swaps and random ones. The fold swap has no reference past its fold."""
function swap_cases()::AbstractVector{BenchmarkCase}
    return BenchmarkCase[
        swap_case("near swap", NEAR_SWAP, widened(NEAR_SWAP.before), widened(NEAR_SWAP.after)),
        swap_case("far swap", FAR_SWAP, widened(FAR_SWAP.before), widened(FAR_SWAP.after)),
        map(swap_case, 1:6)...,
    ]
end

"""A logger that sends each trace or redraw-crossing record's message and `reason` (`:none` without one) into `sink`."""
struct TraceCounter <: AbstractLogger
    sink::Channel{Pair{String, Symbol}}
end

min_enabled_level(::TraceCounter)::typeof(Debug) = Debug
shouldlog(::TraceCounter, ::LogLevel, ::Module, group::Any, ::Any)::Bool = group in (:trace, :homotopy_crossing)
catch_exceptions(::TraceCounter)::Bool = false

function handle_message(counter::TraceCounter, ::LogLevel, message::Any, ::Module, ::Any, ::Any, ::Any, ::Any; keywords...)::Nothing
    put!(counter.sink, string(message) => Symbol(get(keywords, :reason, :none)))
    return nothing
end

# Checks GameTracing's corrector callback logs when it stops or refuses a corrector; its `:contraction` and
# `:precision_exhausted` stops are not counted, and BifurcationKit's distance refusals show only as halved steps.
const REFUSAL_CHECKS = (:reach, :kl, :angle, :nonfinite)
const REFUSAL_MESSAGES = ("corrector refused", "corrector stopped")

"""`case` traced under `settings`; a `SwapCase` redraws its map through `SwapAt`. A `Function`, as `with_logger` requires."""
struct TraceRun{C, S <: TraceSettings} <: Function
    case::C
    settings::S
end

function (trace_run::TraceRun{C})()::GameTracing.TraceResult{Float32} where {C <: Case}
    (; case, settings) = trace_run
    return trace_endpoint(LogitGame(case.pass, case.precision), case.width, settings)
end

function (trace_run::TraceRun{C})()::GameTracing.TraceResult{Float32} where {C <: SwapCase}
    (; settings) = trace_run
    (; case, after, swap_level) = trace_run.case
    return trace_endpoint(LogitGame(case.pass, case.precision), case.width, settings; redraw = SwapAt(LogitGame(after, case.precision), swap_level))
end

"""First-action probabilities of `logits`."""
function first_choices(logits::AbstractVector{E})::AbstractVector{E} where {E <: Real}
    return inv.(one(E) .+ exp.(-logits))
end

"""Counts the refusals in `refusals` that a given check made."""
struct RefusalCount{V <: AbstractVector{Symbol}}
    refusals::V
end

(counter::RefusalCount)(check::Symbol)::Integer = count(==(check), counter.refusals)

"""
What one trace or LogitNash solve did: first-action probabilities (empty on failure), its status, accepted steps,
accepted steps of its redraw crossings, corrector iterates, bytes, refusals per check (`REFUSAL_CHECKS`), the check
that refused last and the reason it stopped short (`:none` if none).
"""
struct Outcome
    choices::Vector{Float32}
    status::String
    steps::Int
    crossing_steps::Int
    iterations::Int
    bytes::Int
    refusals::NTuple{length(REFUSAL_CHECKS), Int}
    last_refusal::Symbol
    stop_reason::Symbol
end

"""`case` traced under `settings` (`TraceRun`), with its bytes and logged records."""
function Outcome(case::C, settings::S)::Outcome where {C <: BenchmarkCase, S <: TraceSettings}
    trace_run = TraceRun(case, settings)
    trace_run()
    bytes = @allocated trace_run()
    sink = Channel{Pair{String, Symbol}}(typemax(Int))
    return Outcome(with_logger(trace_run, TraceCounter(sink)), sink, bytes)
end

"""The `Outcome` of a trace that returned `result`, allocated `bytes` and logged into `sink`, which this closes."""
function Outcome(result::R, sink::AbstractChannel{Pair{String, Symbol}}, bytes::Integer)::Outcome where {R <: Result}
    close(sink)
    records = collect(sink)
    refusals = [last(record) for record in records if first(record) in REFUSAL_MESSAGES]
    stops = [last(record) for record in records if first(record) == "trace stopped"]
    tally = (
        count(==("trace point") ∘ first, records), count(==("crossing point") ∘ first, records),
        count(==("corrector iterate") ∘ first, records), bytes, map(RefusalCount(refusals), REFUSAL_CHECKS),
        last([:none; refusals]), last([:none; stops]),
    )
    if is_error(result)
        return Outcome(Float32[], string(unwrap_error(result).cause), tally...)
    end
    return Outcome(first_choices(unwrap(result).logits), "reached", tally...)
end

"""
The largest probability gap between `left` and `right`, NaN when either is empty, in `right`'s element type: against
a Float64 reference that resolves gaps below Float32's 6e-8 spacing near 1.
"""
function profile_gap(left::AbstractVector, right::AbstractVector{E})::E where {E <: Real}
    if isempty(left) || isempty(right)
        return E(NaN32)
    end
    return maximum(abs, E.(left) .- right)
end

"""One variant's `outcome` on game `case`: its reference kind, whether the game is a pitchfork or a swap, and the gap to that reference."""
struct Row
    case::String
    label::String
    outcome::Outcome
    kind::String
    pitchfork::Bool
    swapped::Bool
    gap::Float32
end

"""Variant `label`'s row on `case`, with `outcome` at `gap` from `reference`."""
function Row(case::C, label::AbstractString, outcome::Outcome, reference::NamedTuple, gap::Real)::Row where {C <: BenchmarkCase}
    return Row(case_name(case), label, outcome, reference_kind(reference), Symbol(case_name(case)) in PITCHFORK_GAMES, case isa SwapCase, gap)
end

"""The row of variant `label => settings` on `case`, whose shared reference is `shared` (`shared_reference`)."""
function Row(case::C, shared::H, (label, settings)::Pair{String, S})::Row where {C <: BenchmarkCase, H, S <: TraceSettings}
    outcome = Outcome(case, settings)
    reference = variant_reference(case, settings, shared)
    # A folded branch has no reference: its gap is NaN.
    return Row(case, label, outcome, reference, profile_gap(ifelse(reference.reached, outcome.choices, empty(outcome.choices)), reference.first_choices))
end

"""LogitNash's solve of `payoffs` to `precision`, with no regret stop: `(profile, status)`."""
struct LogitNashRun{P <: Tuple, E <: Real}
    payoffs::P
    precision::E
end

(solve::LogitNashRun)()::Tuple = LogitNash.solve(solve.payoffs; stop_lambda = solve.precision, stop_eps = zero(solve.precision))

"""LogitNash's row on `case` from `logitnash_run`, its bytes from a second run after the first compiles it."""
function Row(logitnash_run::L, case::C)::Row where {L <: LogitNashRun, C <: Case}
    logitnash_run()
    bytes = @allocated logitnash_run()
    return Row(case, logitnash_run(), bytes)
end

"""
LogitNash's row on `case` from its `(profile, status)`: its iterations as steps, and no corrector iterates or refusals,
which it does not report. LogitNash stops past the target, so its gap is against a reference at the precision it stopped at.
"""
function Row(case::C, (profile, status)::T, bytes::Integer)::Row where {C <: Case, T <: Tuple}
    choices = [first(strategy) for strategy in profile]
    reached = !status.stall && status.lambda >= case.precision
    reference = branch_reference(case.reference_pass, case.width, status.lambda)
    outcome = Outcome(
        ifelse(reached, choices, empty(choices)), ifelse(reached, "reached", "stalled"), status.iteration, 0, 0, bytes,
        map(Returns(0), REFUSAL_CHECKS), :none, ifelse(reached, :none, :stall),
    )
    return Row(case, "logitnash", outcome, reference, profile_gap(ifelse(reached && reference.reached, choices, empty(choices)), reference.first_choices))
end

"""Whether LogitNash runs on `case`: a fixed-map normal-form game, the only kind it takes."""
logitnash_supported(::BenchmarkCase)::Bool = false
logitnash_supported(::Case{<:Any, <:NormalFormPass})::Bool = true

"""`case`'s LogitNash row, or none where LogitNash does not run (`logitnash_supported`)."""
logitnash_rows(::BenchmarkCase)::AbstractVector{Row} = Row[]
function logitnash_rows(case::Case{<:Any, <:NormalFormPass})::AbstractVector{Row}
    return [Row(LogitNashRun(case.reference_pass.payoffs, case.precision * float(1)), case)]
end

"""Whether `row`'s trace reached its target."""
reached_row(row::Row)::Bool = row.outcome.status == "reached"

"""Whether `row` has an endpoint to compare."""
has_choices(row::Row)::Bool = !isempty(row.outcome.choices)

"""Whether `row` comes from variant `label`."""
has_label(row::Any, label::AbstractString)::Bool = row.label == label

"""Whether `row` is on a map-swap game exactly when `swapped`."""
in_group(row::Row, swapped::Bool)::Bool = row.swapped == swapped

"""`row` as a printed ROW line."""
function row_line(row::Row)::AbstractString
    (; outcome) = row
    return @sprintf(
        "ROW | %-24s | %-14s | %-13s | steps %5d | crossing %4d | iterations %6d | refused reach %d kl %d angle %d nonfinite %d, last %s | stopped %s | gap %.2e (%s) | bytes %d\n",
        row.case, row.label, outcome.status, outcome.steps, outcome.crossing_steps, outcome.iterations, outcome.refusals...,
        outcome.last_refusal, outcome.stop_reason, row.gap, row.kind, outcome.bytes,
    )
end

"""`row` as a CSV line."""
function csv_line(row::Row)::AbstractString
    (; outcome) = row
    fields = (
        row.case, row.label, outcome.status, outcome.steps, outcome.crossing_steps, outcome.iterations, outcome.refusals...,
        outcome.last_refusal, outcome.stop_reason, row.gap, outcome.bytes, row.kind, row.swapped,
    )
    return string(join(fields, ','), '\n')
end

const CSV_HEADER = "game,variant,status,steps,crossing_steps,iterations,refused_reach,refused_kl,refused_angle,refused_nonfinite,last_refusal,stop_reason,gap,bytes,reference,swapped\n"

"""Whether the endpoints of the row pair `rows` lie over 1e-3 apart."""
diverged(rows::Tuple)::Bool = profile_gap(first(rows).outcome.choices, last(rows).outcome.choices) > 1.0f-3

"""The SIGNAL line of the row pair `rows`."""
function signal_line(rows::T)::AbstractString where {T <: Tuple}
    left, right = rows
    return @sprintf("SIGNAL | %s | %s vs %s | %.3e\n", left.case, left.label, right.label, profile_gap(left.outcome.choices, right.outcome.choices))
end

"""A SIGNAL line for each pair of `reached` rows whose endpoints lie over 1e-3 apart."""
function signal_lines(reached::V)::AbstractVector{<:AbstractString} where {V <: AbstractVector{Row}}
    ordered = [(left, right) for (left_index, left) in pairs(reached), (right_index, right) in pairs(reached) if left_index < right_index]
    return map(signal_line, filter(diverged, ordered))
end

"""
Every variant's row on `case` and LogitNash's (`logitnash_rows`), printed to `io`, with a SIGNAL for each pair of
traced endpoints over 1e-3 apart.
"""
function case_rows(case::C, variants::V, io::IO)::AbstractVector{Row} where {C <: BenchmarkCase, V <: AbstractVector}
    traced = Row.((case,), (shared_reference(case),), variants)
    rows = vcat(traced, logitnash_rows(case))
    print(io, join(map(row_line, rows)), join(signal_lines(filter(has_choices, traced))))
    flush(io)
    return rows
end

"""`reduction` of `values`, NaN when there are none."""
function statistic(reduction::Any, values::AbstractVector{E})::E where {E <: Real}
    if isempty(values)
        return E(NaN32)
    end
    return reduction(values)
end

"""
The SUMMARY line for variant `label` over the `rows` of one game group (`group` names it); the gap statistics skip
pitchfork games, whose branch is ambiguous. Failures list each stall's stop reason and the check that refused last.
"""
function summary_line(group::AbstractString, label::AbstractString, rows::V)::AbstractString where {V <: AbstractVector{Row}}
    mine = filter(Base.Fix2(has_label, label), rows)
    gaps = [row.gap for row in mine if reached_row(row) && !isnan(row.gap) && !row.pitchfork]
    failures = join(sort([string(row.outcome.status, ":", row.outcome.stop_reason, ":", row.outcome.last_refusal) for row in mine if !reached_row(row)]), " ")
    return @sprintf(
        "SUMMARY %-5s | %-14s | reached %2d/%2d | off-reference %2d | steps %6d | crossing %5d | iterations %7d | gap median %.1e worst %.1e | bytes mean %d | failed %s\n",
        group, label, count(reached_row, mine), length(mine), count(>(1.0f-3), gaps),
        sum(row.outcome.steps for row in mine), sum(row.outcome.crossing_steps for row in mine), sum(row.outcome.iterations for row in mine),
        statistic(median, gaps), statistic(maximum, gaps), round(Int, sum(row.outcome.bytes for row in mine) / length(mine)), failures,
    )
end

# The unfolded pitchforks: battles of the sexes with every payoff moved by Gaussian noise at each scale.
const UNFOLDING_SCALES = inv.(Float32.((10^4, 10^3, 10^2, 10)))
const UNFOLDING_DRAWS = 40
# The same games every run, so two runs on one CPU compare row by row.
const UNFOLDING_SEED = 20261029

"""Moves a payoff table by Gaussian noise of scale `scale` drawn from `rng`."""
struct ScaledNoise{R <: AbstractRNG, E <: Real}
    rng::R
    scale::E
end

(noise::ScaledNoise)(table::AbstractArray)::AbstractArray = table .+ noise.scale .* randn(noise.rng, size(table)...)

"""
One variant's trace of an unfolded battle of the sexes: whether it ended on the principal branch (the arclength
reference's endpoint, to 1e-3), the reference's crossings of a singular point and the trace's steps.
"""
struct UnfoldedRow
    scale::Float32
    draw::Int
    label::String
    status::String
    steps::Int
    crossings::Int
    principal::Bool
end

"""The row of variant `label => settings` on draw `draw` at noise `scale`: game `case` with reference `reference`."""
function UnfoldedRow(scale::Real, draw::Integer, case::C, reference::NamedTuple, (label, settings)::Pair{String, S})::UnfoldedRow where {C <: Case, S <: TraceSettings}
    outcome = Outcome(case, settings)
    principal = reference.reached && profile_gap(outcome.choices, reference.first_choices) < 1.0f-3
    return UnfoldedRow(scale, draw, label, outcome.status, outcome.steps, reference.crossings, principal)
end

reached_row(row::UnfoldedRow)::Bool = row.status == "reached"

"""Whether `row`'s trace ended on the principal branch."""
principal_row(row::UnfoldedRow)::Bool = row.principal

"""Whether `row` is at noise scale `scale`."""
has_scale(row::UnfoldedRow, scale::Real)::Bool = row.scale == scale

"""One unfolded battle of the sexes at noise `scale` from `rng`, traced under each of `variants` to precision 4."""
function unfolded_rows(rng::G, scale::Real, draw::Integer, variants::V)::AbstractVector{UnfoldedRow} where {G <: AbstractRNG, V <: AbstractVector}
    payoffs = map(ScaledNoise(rng, scale), map(float, LOGITNASH_GAMES.battle_of_the_sexes))
    case = normal_form_case(string("unfolded ", scale, " draw ", draw), payoffs, 4)
    return UnfoldedRow.((scale,), (draw,), (case,), (case_reference(case),), variants)
end

"""`row` as a CSV line."""
csv_line(row::UnfoldedRow)::AbstractString = string(join((row.scale, row.draw, row.label, row.status, row.steps, row.crossings, row.principal), ','), '\n')

"""The PITCHFORK line of variant `label` over its `rows` at noise `scale`: draws on the principal branch, reference crossings and steps."""
function pitchfork_line(scale::Real, label::AbstractString, rows::V)::AbstractString where {V <: AbstractVector{UnfoldedRow}}
    return @sprintf(
        "PITCHFORK | scale %.0e | %-30s | principal %2d/%2d | reached %2d | reference crossings %d | steps %5d\n",
        scale, label, count(principal_row, rows), length(rows), count(reached_row, rows),
        sum(row.crossings for row in rows), sum(row.steps for row in rows),
    )
end

"""Whether `variant` runs in the pitchfork benchmark: the shipped settings and fisher_reach."""
pitchfork_variant(variant::Pair)::Bool = first(variant) == "shipped" || startswith(first(variant), "fisher_reach ")

"""
The shipped settings `shipped` and every arm (`arm_variants`) as `label => settings` pairs of one settings type,
which an arm of another type fails to convert to; `"quick"` keeps the shipped settings and the first arm.
"""
function benchmark_variants(scope::AbstractString, shipped::S)::AbstractVector{Pair{String, S}} where {S <: TraceSettings}
    variants = Pair{String, S}["shipped" => shipped; arm_variants()]
    if scope == "quick"
        return variants[1:2]
    end
    return variants
end

"""
The unfolded-pitchfork benchmark: `draws` draws per scale from `seed` under the shipped settings and fisher_reach,
rows to `output`, and a PITCHFORK line per scale and variant to `io`.
"""
function pitchfork_benchmark(output::AbstractString, io::IO, draws::Integer, seed::Integer)::AbstractVector{UnfoldedRow}
    println(io, "SEED | ", seed)
    variants = filter(pitchfork_variant, benchmark_variants("full", TraceSettings{Float32}()))
    # Scales run along the second axis, so the draws come scale by scale, in the order the rows list them.
    rows::Vector{UnfoldedRow} = reduce(vcat, vec(unfolded_rows.((Xoshiro(seed),), permutedims(collect(UNFOLDING_SCALES)), 1:draws, (variants,))))
    write(output, string("scale,draw,variant,status,steps,crossings,principal\n", join(map(csv_line, rows))))
    for scale in UNFOLDING_SCALES, label in first.(variants)
        print(io, pitchfork_line(scale, label, filter(Base.Fix2(has_label, label), filter(Base.Fix2(has_scale, scale), rows))))
    end
    return rows
end

"""The games of `scope`: `"full"` every one, `"quick"` one fixed and one swapped game."""
function scope_cases(scope::AbstractString)::AbstractVector{BenchmarkCase}
    if scope == "quick"
        return BenchmarkCase[first(fixed_cases()), first(swap_cases())]
    elseif scope == "full"
        return vcat(fixed_cases(), swap_cases())
    end
    throw(ArgumentError("scope must be \"full\", \"quick\" or \"pitchfork\", got $(repr(scope))"))
end

"""
Every row of `cases` under `variants`, printed to `io` with a SUMMARY per group and variant, and written to `output`.
Throws unless every game and variant gave a row.
"""
function benchmark_rows(output::AbstractString, cases::C, variants::V, io::IO)::AbstractVector{Row} where {C <: AbstractVector, V <: AbstractVector}
    rows::Vector{Row} = reduce(vcat, case_rows.(cases, (variants,), (io,)))
    expected = length(cases) * length(variants) + count(logitnash_supported, cases)
    if length(rows) != expected
        error("expected $expected rows, got $(length(rows))")
    end
    write(output, string(CSV_HEADER, join(map(csv_line, rows))))
    # Fixed-map and map-swap games are summarized apart: a swap's refusals come from the redraw, not the geometry.
    for (group, swapped) in (("fixed", false), ("swap", true)), label in [first.(variants); ifelse(swapped, String[], ["logitnash"])]
        print(io, summary_line(group, label, filter(Base.Fix2(in_group, swapped), rows)))
    end
    return rows
end

"""
Run the benchmark over `scope`, printing to `io`: `"full"` every game and variant, `"quick"` one fixed and one swapped
game under the shipped settings and the first arm, the load check CI runs; `"pitchfork"` the unfolded pitchforks
(`pitchfork_benchmark`, `UNFOLDING_DRAWS` draws per scale from `UNFOLDING_SEED`). Returns the rows it wrote to `output`.
"""
function main(output::AbstractString, scope::AbstractString, io::IO)::AbstractVector
    # Step counts can differ between runners; the CPU and its LLVM target name the hardware each run used.
    println(io, "RUNNER | ", first(Sys.cpu_info()).model, " | target ", Sys.CPU_NAME, " | threads ", Threads.nthreads(), " | Julia ", VERSION)
    if scope == "pitchfork"
        return pitchfork_benchmark(output, io, UNFOLDING_DRAWS, UNFOLDING_SEED)
    end
    return benchmark_rows(output, scope_cases(scope), benchmark_variants(scope, TraceSettings{Float32}()), io)
end

# Run as a script; the code rules load this file as a module and run `main` themselves.
if abspath(PROGRAM_FILE) == @__FILE__
    main(get(ARGS, 1, joinpath(@__DIR__, "arms.csv")), get(ARGS, 2, "full"), stdout)
end
