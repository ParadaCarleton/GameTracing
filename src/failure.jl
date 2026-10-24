"""Why a trace stopped short of its target precision."""
@enum TraceCause NoConvergence TraceStuck NotFinite Singular

"""
    TraceFailure(cause, context)

A trace that did not reach its target: `cause` names why and `context` carries the last point,
tangent and step the trace held.
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

"""A trace's outcome: `Traced`, or the `TraceFailure` that stopped it, whose context holds the last accepted point."""
const TraceResult{E <: Real} = Union{Traced{E}, TraceFailure}

"""The value of `x` with any automatic-differentiation tangent dropped; clients extend it for their dual types."""
primal(x::Real)::typeof(x) = x

"""
    uncorrected_start(exception)

Whether `exception` is BifurcationKit's report that a start point would not correct: a bare
`"Stopping continuation."` at the first point, an `ErrorException` at the second.
"""
function uncorrected_start(@nospecialize(exception::Any))::Bool
    if exception isa String
        return exception == "Stopping continuation."
    end
    if exception isa ErrorException
        return startswith(exception.msg, "Newton failed to converge")
    end
    return false
end

"""
    solver_blowup(exception)

Whether `exception` is a broken linear solve or non-finite factorization: `SingularException`,
`LAPACKException`, or LAPACK's `ArgumentError` for a NaN or infinite matrix. A bare `ArgumentError`
is a caller's contract violation and propagates.
"""
function solver_blowup(@nospecialize(exception::Any))::Bool
    if exception isa SingularException || exception isa LAPACKException
        return true
    end
    if exception isa ArgumentError
        message = String(exception.msg)
        return contains(message, "LAPACK") ||
            contains(message, "matrix contains Infs or NaNs")
    end
    return false
end
