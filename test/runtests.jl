using GameTracing
using Test
using ErrorTypes: is_error, unwrap, unwrap_error

include("games.jl")
include("fixtures.jl")

# SCRATCH: every accepted point of the BorderingSolve fold arm, with and without the residual_scale floor; not for landing.
struct UnscaledPass{P}
    pass::P
end
(pass::UnscaledPass)(arguments...)::Nothing = pass.pass(arguments...)
GameTracing.gain_scale(::UnscaledPass, target_precision::T)::T where {T <: Real} = zero(T)

struct PointPrinter{E <: Real}
    label::String
    target::E
end
function (printer::PointPrinter)(logits::AbstractVector, relative_precision::Real)::Bool
    println(printer.label, " λ=", relative_precision * printer.target, " logits=", string(logits))
    return true
end

function main()::Nothing
    settings = TraceSettings{Float32}(bordered_solver = BorderingSolve)
    for (label, pass) in (("scaled", FOLD_GAME), ("unscaled", UnscaledPass(FOLD_GAME)))
        result = trace_endpoint(LogitGame(pass, 2.0f0), 1, settings; keep_going = PointPrinter(label, 2.0f0))
        println(label, " result: ", is_error(result) ? unwrap_error(result) : unwrap(result))
    end
    return nothing
end
main()
