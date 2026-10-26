"""
    GameTracing

Homotopy continuation for logit quantal response equilibria. `trace_endpoint` follows the
branch of a logit game from zero precision (infinite temperature), where the only equilibrium is the flat one,
up to a target precision with BifurcationKit's pseudo-arclength continuation (PALC), across folds.
Failures are values (`TraceFailure`), never exceptions.
"""
module GameTracing

export LogitGame,
    TraceSettings,
    TraceCause,
    TraceFailure,
    Traced,
    TraceResult,
    StepControl,
    CorrectorSettings,
    TangentMethod,
    BorderedTangent,
    SecantTangent,
    BorderedSolver,
    MatrixSolve,
    BorderingSolve,
    trace_endpoint,
    refresh_kl,
    refresh_verdict

include("failure.jl")
include("bordered.jl")
include("branch.jl")
include("callbacks.jl")
include("trace.jl")
include("refresh.jl")

end
