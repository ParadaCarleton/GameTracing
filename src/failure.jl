using LinearAlgebra: LAPACKException, SingularException
"""Why a trace stopped short of its target precision."""
@enum TraceCause NoConvergence TraceStuck NotFinite Singular

"""
    TraceFailure(cause, context)

A trace that did not reach its target: `cause` names why and `context` carries the last accepted `logits`
and relative precision. A trace that left its start also carries the tangent there in logits (`logit_tangent`,
`dz = cosh(u) du`) and the step it held; a non-finite start has neither.
"""
struct TraceFailure{C <: NamedTuple}
    cause::TraceCause
    context::C
end

"""
    Traced(logits)

The `logits` of a trace at its target precision.
"""
struct Traced{E <: Real}
    logits::Vector{E}
end

"""A trace's outcome: `Traced`, or the `TraceFailure` that stopped it, whose context holds the last accepted `logits`."""
const TraceResult{E <: Real} = Union{Traced{E}, TraceFailure}

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
