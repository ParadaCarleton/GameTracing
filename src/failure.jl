using ErrorTypes: Result
using LinearAlgebra: LAPACKException, SingularException
"""
Why a trace stopped short of its target precision. `PrecisionExhausted`: Newton's corrections stopped contracting at
the float type's rounding noise, before the target (`CorrectorReach`); a wider float type goes further.
"""
@enum TraceCause NoConvergence TraceStuck NotFinite Singular PrecisionExhausted

"""
    TraceFailure(cause, context)

A trace that did not reach its target: `cause` names why and `context` carries the last accepted `logits`
and relative precision. A trace that left its start also carries the tangent there in logits (`logit_tangent`,
`dz = ρ dw + w dρ` in the payoff gaps `w`) and the step it held; a non-finite start has neither.
"""
struct TraceFailure{C <: NamedTuple}
    cause::TraceCause
    context::C
end

"""
    Traced(logits, steps, logit_accuracy)

The `logits` of a trace at its target precision, a root of the game in effect there, the continuation `steps` it took
over every segment, and `logit_accuracy`: the largest last Newton correction in logits of any accepted point, at most
the corrector's `logit_tolerance`.
"""
struct Traced{E <: Real}
    logits::Vector{E}
    steps::Int
    logit_accuracy::E
end

"""
A trace's outcome, an ErrorTypes `Result`: `Ok(Traced)`, or `Err(TraceFailure)` whose context holds the last
accepted `logits`. A caller reaches the logits through `unwrap`, `@?` or a check of `is_error`, never by field
access on the result; `unwrap` on a failure throws at the caller.
"""
const TraceResult{E <: Real} = Result{Traced{E}, TraceFailure}

"""
    uncorrected_start(exception)

Whether `exception` is BifurcationKit's report that a start point would not correct: a bare
`"Stopping continuation."` at the first point, an `ErrorException` at the second.
"""
uncorrected_start(::Any)::Bool = false
uncorrected_start(message::String)::Bool = message == "Stopping continuation."
uncorrected_start(exception::ErrorException)::Bool = startswith(
    exception.msg,
    "Newton failed to converge",
)

"""
    solver_blowup(exception)

Whether `exception` is a broken linear solve or non-finite factorization: `SingularException`,
`LAPACKException`, or LAPACK's `ArgumentError` for a NaN or infinite matrix. A bare `ArgumentError`
is a caller's contract violation and propagates.
"""
solver_blowup(::Any)::Bool = false
solver_blowup(::SingularException)::Bool = true
solver_blowup(::LAPACKException)::Bool = true
function solver_blowup(exception::ArgumentError)::Bool
    return contains(exception.msg, "LAPACK") ||
        contains(exception.msg, "matrix contains Infs or NaNs")
end
