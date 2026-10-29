# EXPERIMENT: which check halves the noisy flat gain's trace to a zero step just past its noise onset.
using ErrorTypes: is_error, unwrap_error
using Logging: Debug
using Printf: @sprintf
using Test: collect_test_logs

const NOISY_TARGET = 1.0f5
const NOISY_SETTINGS = TraceSettings{Float32}(;
    geometry = TraceGeometry{Float32}(; distance = FisherLogPrecisionDistance), max_steps = 20_000,
)
const TAIL_RECORDS = 80

"""The noisy flat gain traced to `NOISY_TARGET`, a `Function` as `collect_test_logs` takes."""
struct NoisyTrace{G} <: Function
    game::G
end

(trace::NoisyTrace)()::TraceResult{Float32} = trace_endpoint(trace.game, 1, NOISY_SETTINGS)

"""One log record's message and keyword arguments on one line."""
function record_line(record::Any)::String
    fields = join([string(name, "=", entry) for (name, entry) in pairs(record.kwargs) if name != :state], " ")
    return string(record.message, " ", fields)
end

"""The precision `λ` after `record`: its relative precision times the target if it carries one, else `current`."""
function carried_precision(current::E, record::Any)::E where {E <: Real}
    if haskey(record.kwargs, :relative_precision)
        return E(record.kwargs[:relative_precision] * NOISY_TARGET)
    end
    return current
end

"""Each record of `records` with its precision, the first `count` from `from` on, one line each, labelled `label`."""
function print_records(label::AbstractString, records::AbstractVector, precisions::AbstractVector, from::Integer, count::Integer)::Nothing
    for (record, precision) in Iterators.take(Iterators.drop(zip(records, precisions), from - 1), count)
        println("NOISY ", label, " λ=", precision, " ", record_line(record))
    end
    return nothing
end

@testset "EXPERIMENT: the noisy flat gain's stall past its noise onset" begin
    pass = NoisyFlatPass(1.0f-4, 1.0f-7, 10.0f0)
    logs, result = collect_test_logs(NoisyTrace(LogitGame(pass, NOISY_TARGET)); min_level = Debug)
    records = [record for record in logs if record.group == :trace]
    precisions = accumulate(carried_precision, records; init = zero(Float32))
    cause = ifelse(is_error(result), string(unwrap_error(result).cause), "reached")
    println("NOISY outcome: ", cause, ", ", length(records), " trace records")
    for (label, window) in (("below onset", precisions .< 10), ("past onset", precisions .>= 10))
        stops = [
            string(record.message, ":", get(record.kwargs, :reason, get(record.kwargs, :converged, "")))
            for (record, inside) in zip(records, window) if inside && record.message != "corrector iterate" && record.message != "corrector correction"
        ]
        tally = sort([(count(==(kind), stops), kind) for kind in unique(stops)]; rev = true)
        println("NOISY ", label, ": ", join([@sprintf("%s ×%d", kind, number) for (number, kind) in tally], ", "))
    end
    past = [record for (record, precision) in zip(records, precisions) if precision >= 10 && record.message == "corrector correction"]
    corrections = sort([Float32(record.kwargs[:correction]) for record in past])
    pins = sort([abs(Float32(record.kwargs[:pin])) for record in past])
    if !isempty(corrections)
        println(@sprintf("NOISY past onset: %d corrections, min/median/max %.3g / %.3g / %.3g; abs pin min/median/max %.3g / %.3g / %.3g",
            length(corrections), first(corrections), corrections[cld(end, 2)], last(corrections),
            first(pins), pins[cld(end, 2)], last(pins)))
    end
    first_past = something(findfirst(>=(10), precisions), length(records) + 1)
    print_records("head", records, precisions, first_past, TAIL_RECORDS)
    print_records("tail", records, precisions, max(1, length(records) - TAIL_RECORDS + 1), TAIL_RECORDS)
    @test !isempty(records)
end
