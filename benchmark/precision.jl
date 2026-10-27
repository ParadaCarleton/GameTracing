# Separates step size from element type on the unfolded pitchfork: traces it at Float32 and Float64, at the default
# and at a 1e-5 largest step, through ρ ∈ [0.14, 0.2], and logs the branch gap and the corrector at the departure.
using GameTracing
using ErrorTypes: is_error, unwrap, unwrap_error
using LinearAlgebra: norm
using Logging: Logging, AbstractLogger, Debug, LogLevel, with_logger
using LogitNash: LogitNash
using Random: Xoshiro

const TEST_DIRECTORY = joinpath(@__DIR__, "..", "test")
include(joinpath(TEST_DIRECTORY, "games.jl"))

# trace-arms' UNFOLDED_PITCHFORK_GAME.
const GAME_PAYOFFS = ([30000726 -37; 757 20000821] .// 10^7, [19999477 -283; 919 29999523] .// 10^7)
const STOP_PRECISION = 0.2
const CAPACITY = 400_000
const PROBES = (0.14, 0.15, 0.16, 0.163, 0.165, 0.17, 0.18, 0.19)

"""Accepted points `[logits; ρ]`, one per column, unfilled columns NaN; stops the trace past `stop`."""
struct PathRecorder{M <: AbstractMatrix, E <: Real}
    points::M
    stop::E
end

function (recorder::PathRecorder)(logits::AbstractVector{E}, relative_precision::E)::Bool where {E <: Real}
    column = findfirst(isnan, view(recorder.points, 1, :))
    if column !== nothing
        recorder.points[:, column] .= [logits; relative_precision]
    end
    return relative_precision < recorder.stop
end

"""Prints the trace's corrector messages while the last relative precision it logged lies in `[low, high]`."""
struct WindowLogger{V <: AbstractVector} <: AbstractLogger
    low::Float64
    high::Float64
    # The last relative precision a corrector iterate logged; a diagnostic's own scratch element.
    last_precision::V
end

Logging.min_enabled_level(::WindowLogger)::LogLevel = Debug
Logging.shouldlog(::WindowLogger, level::LogLevel, _module::Module, group::Any, id::Any)::Bool = group == :trace
Logging.catch_exceptions(::WindowLogger)::Bool = false

function Logging.handle_message(logger::WindowLogger, level::LogLevel, message::Any, _module::Module, group::Any, id::Any, file::Any, line::Any; keywords...)::Nothing
    values = Dict(keywords)
    if haskey(values, :relative_precision)
        logger.last_precision[1] = Float64(values[:relative_precision])
    end
    if logger.low <= logger.last_precision[1] <= logger.high
        println("CORRECTOR ", message, " ", join(("$key=$value" for (key, value) in values), " "))
    end
    return nothing
end

"""Every root of `pass` at precision `level` that Newton reaches from a grid of logit starts, without duplicates."""
function roots_at(pass::Any, level::Real)::Matrix{Float64}
    starts = [[first, second] for first in -3:0.25:3 for second in -3:0.25:3]
    found = fill(NaN, 2, length(starts))
    for (column, start) in pairs(starts)
        root = start .* float(1)
        for _ in 1:60
            gains = evaluated(pass, root, level)
            root = root .- (one(gains.logit_slope) .- gains.logit_slope) \ (root .- gains.response)
        end
        if residual_norm(pass, root, level) < 1.0e-12 && !any(norm(root .- found[:, other]) < 1.0e-7 for other in 1:(column - 1) if !isnan(found[1, other]))
            found[:, column] .= root
        end
    end
    return found[:, findall(!isnan, view(found, 1, :))]
end

"""Log the branch structure at each probe precision: the reference root and its nearest other root."""
function report_branches(pass::Any, precision::Real)::Nothing
    for probe in PROBES
        level = probe * precision
        reference = branch_reference(pass, 2, level)
        roots = roots_at(pass, level)
        distances = [norm(root .- reference.logits) for root in eachcol(roots)]
        others = filter(>(1.0e-7), distances)
        println("BRANCHES ρ $probe λ $(round(level; digits = 5)) roots $(size(roots, 2)) reference $(round.(reference.logits; digits = 6)) ",
            "nearest_other_logit_distance $(isempty(others) ? NaN : minimum(others))")
    end
    return nothing
end

"""The first accepted point farther than `threshold` in logits from the reference, as `(; index, precision)`, or nothing."""
function departure(pass::Any, points::AbstractMatrix, precision::Real, threshold::Real)::NamedTuple
    filled = count(!isnan, view(points, 1, :))
    for (index, point) in enumerate(eachcol(view(points, :, 1:filled)))
        relative = Float64(point[end])
        if 0.14 <= relative
            reference = branch_reference(pass, 2, relative * precision)
            gap = norm(Float64.(point[1:2]) .- reference.logits)
            if gap > threshold
                return (; index, precision = relative, gap)
            end
        end
    end
    return (; index = 0, precision = NaN, gap = NaN)
end

function run_arm(label::AbstractString, ::Type{E}, max_step::Real, pass64::Any, precision::Real)::Nothing where {E <: Real}
    settings = TraceSettings{E}(; first_step = E(min(0.05, max_step)), max_step = E(max_step), max_steps = 1_000_000)
    game = LogitGame(NormalFormPass(map(table -> E.(table), GAME_PAYOFFS)), E(precision))
    recorder = PathRecorder(fill(E(NaN), 3, CAPACITY), E(STOP_PRECISION))
    result = trace_endpoint(game, 2, settings; keep_going = recorder)
    filled = count(!isnan, view(recorder.points, 1, :))
    outcome = is_error(result) ? string(unwrap_error(result).cause) : "reached"
    last_point = Float64.(recorder.points[:, filled])
    reference = branch_reference(pass64, 2, last_point[end] * precision)
    final_gap = norm(last_point[1:2] .- reference.logits)
    sample = recorder.points[:, max(filled ÷ 2, 1)]
    tolerance = GameTracing.corrector_tolerance(settings.corrector.residual_tolerance, settings.corrector.tolerance_floor_ulps, sample)
    floor = settings.corrector.tolerance_floor_ulps * eps(E) * maximum(abs, sample[1:2] ./ sample[3])
    println("ARM $label type $E max_step $max_step outcome $outcome accepted $filled last_ρ $(round(last_point[end]; digits = 5)) ",
        "gap_at_last $final_gap tolerance $tolerance floor $floor jumped $(final_gap > 1.0e-3)")
    leaving = departure(pass64, recorder.points, precision, 1.0e-4)
    println("DEPARTURE $label index $(leaving.index) ρ $(leaving.precision) gap $(leaving.gap)")
    if leaving.index > 1
        before = recorder.points[:, leaving.index - 1]
        println("DEPARTURE_STEP $label from $(Float64.(before)) to $(Float64.(recorder.points[:, leaving.index]))")
        window = WindowLogger(Float64(before[end]), Float64(recorder.points[end, leaving.index]), [NaN])
        with_logger(window) do
            trace_endpoint(game, 2, settings; keep_going = PathRecorder(fill(E(NaN), 3, CAPACITY), E(leaving.precision + 1.0e-9)))
        end
    end
    return nothing
end

function main()::Nothing
    _, status = LogitNash.solve(map(float, GAME_PAYOFFS); stop_lambda = 4.0, stop_eps = 0.0)
    precision = status.lambda
    pass64 = NormalFormPass(map(float, GAME_PAYOFFS))
    println("TARGET precision $precision")
    report_branches(pass64, precision)
    run_arm("E Float32 default", Float32, 1000, pass64, precision)
    run_arm("A Float64 default", Float64, 1000, pass64, precision)
    run_arm("B Float32 small", Float32, 1.0e-5, pass64, precision)
    run_arm("C Float64 small", Float64, 1.0e-5, pass64, precision)
    return nothing
end

main()
