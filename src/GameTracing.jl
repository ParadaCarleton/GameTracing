"""
    GameTracing

Homotopy continuation for logit quantal response equilibria. `trace_endpoint` follows the
branch of a logit game from zero precision (infinite temperature), where the only equilibrium is the flat one,
up to a target precision with BifurcationKit's pseudo-arclength continuation (PALC), across folds.
Failures are values (`TraceFailure`), never exceptions.
"""
module GameTracing

export LogitGame,
    NormalFormGains,
    normal_form_game,
    mixed_strategies,
    logit_count,
    TraceSettings,
    TraceCause,
    TraceFailure,
    Traced,
    TraceResult,
    StepControl,
    CorrectorSettings,
    BorderedSolver,
    MatrixSolve,
    BorderingSolve,
    trace_endpoint,
    branch_projection

include("failure.jl")
include("bordered.jl")
include("branch.jl")
include("normal_form.jl")
include("callbacks.jl")
include("trace.jl")
include("projection.jl")

end
