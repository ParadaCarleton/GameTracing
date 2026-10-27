# The LogitNash issue reproducer on the battle of the sexes unfolded by 1e-4, with every LogitNash path step logged.
using LogitNash
using Logging: Logging, AbstractLogger, Debug, LogLevel, with_logger

"""Prints each LogitNash `Path` step's λ and first-action probabilities, copied as they are logged."""
struct PathLogger <: AbstractLogger end

Logging.min_enabled_level(::PathLogger)::LogLevel = Debug
Logging.shouldlog(::PathLogger, level::LogLevel, _module::Module, group::Any, id::Any)::Bool = _module == LogitNash || parentmodule(_module) == LogitNash
Logging.catch_exceptions(::PathLogger)::Bool = false

function Logging.handle_message(::PathLogger, level::LogLevel, message::Any, _module::Module, group::Any, id::Any, file::Any, line::Any; keywords...)::Nothing
    values = Dict(keywords)
    if haskey(values, :pi) && haskey(values, :lambda)
        println("STEP ", get(values, :_id, ""), " λ ", values[:lambda], " π ", round.(first.(values[:pi]); digits = 6))
    end
    return nothing
end

function main()::Nothing
    println("LogitNash ", pkgversion(LogitNash))
    u1 = [30000726 -37; 757 20000821] ./ 10^7
    u2 = [19999477 -283; 919 29999523] ./ 10^7
    pi, status = LogitNash.solve((u1, u2); stop_lambda = 4.0, stop_eps = 0.0)
    println(first.(pi), " ", status.lambda)
    with_logger(PathLogger()) do
        LogitNash.solve((u1, u2); stop_lambda = 4.0, stop_eps = 0.0)
    end
    return nothing
end

main()
