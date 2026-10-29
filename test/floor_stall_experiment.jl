# EXPERIMENT: which check halves the Float32 unfolded-pitchfork trace to a zero step once rounding noise is accepted.
using ErrorTypes: is_error, unwrap_error
using Logging: Debug
using Printf: @sprintf
using Test: collect_test_logs

include("logitnash_tests.jl")

const FLOOR_ARM = "fisher_log_precision_reach state_weight 0.01"
const FLOOR_RECORDS = 150

"""The unfolded pitchfork traced under `settings` to `precision`, a `Function` as `collect_test_logs` takes."""
struct FloorTrace{S, P} <: Function
    settings::S
    precision::P
end

(trace::FloorTrace)()::TraceResult{Float32} = traced_result(UNFOLDED_PITCHFORK_GAME, trace.precision, trace.settings)

"""One log record's message and keyword arguments on one line."""
function floor_line(record::Any)::String
    fields = join([string(name, "=", entry) for (name, entry) in pairs(record.kwargs) if !(name in (:state, :tangent))], " ")
    return string(record.message, " ", fields)
end

@testset "EXPERIMENT: the refusing check past the accepted rounding noise" begin
    precision = logitnash_equilibrium(UNFOLDED_PITCHFORK_GAME, 4).precision
    settings = last(only(filter(variant -> first(variant) == FLOOR_ARM, arm_variants())))
    logs, result = collect_test_logs(FloorTrace(settings, precision); min_level = Debug)
    records = [record for record in logs if record.group == :trace]
    cause = ifelse(is_error(result), string(unwrap_error(result).cause), "reached")
    println("FLOOR outcome: ", cause, ", ", length(records), " trace records")
    first_floor = something(findfirst(record -> get(record.kwargs, :floor, false), records), length(records) + 1)
    println("FLOOR floor active from record ", first_floor)
    for (label, window) in (("before floor", 1:(first_floor - 1)), ("floor active", first_floor:length(records)))
        stops = [
            string(record.message, ":", get(record.kwargs, :reason, get(record.kwargs, :converged, "")))
            for record in view(records, window) if !(record.message in ("corrector iterate", "corrector correction"))
        ]
        tally = sort([(count(==(kind), stops), kind) for kind in unique(stops)]; rev = true)
        println("FLOOR ", label, ": ", join([@sprintf("%s ×%d", kind, number) for (number, kind) in tally], ", "))
    end
    for record in Iterators.drop(records, max(0, first_floor - 40))
        println("FLOOR ", floor_line(record))
    end
    @test !isempty(records)
end
