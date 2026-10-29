# EXPERIMENT: which Float32 part stalls the unfolded-pitchfork trace at λ 0.808; not for landing.
# Arms: A all Float32; B residual and Jacobian in Float64, rounded; C bordered solve in Float64, rounded; D both; E all Float64.
using BifurcationKit: BifurcationKit
using ErrorTypes: is_error, unwrap, unwrap_error
using LinearAlgebra: BlasFloat, BlasInt, LAPACK, LU, SingularException, istril, istriu, logabsdet
using LogitNash: LogitNash
using Logging: Debug
using Printf: @sprintf
using Test: collect_test_logs

const STALL_WINDOW = (0.795f0, 0.82f0)

"""A pass whose branch residual and Jacobian are formed in `widen(E)` from the wide pass `wide`, then rounded to `E`."""
struct WidePass{P}
    wide::P
end

function GameTracing.branch_system!(
    workspace::GameTracing.BranchWorkspace{E},
    game::LogitGame{<:WidePass},
    point::AbstractVector{E},
    pin::Real,
)::typeof(workspace) where {E <: Real}
    wide_game = LogitGame(game.pass!.wide, widen(E)(game.target_precision))
    wide_point = widen(E).(point)
    filled = GameTracing.branch_system!(GameTracing.BranchWorkspace(wide_game, wide_point), wide_game, wide_point, widen(E)(pin))
    workspace.residual .= E.(filled.residual)
    workspace.jacobian .= E.(filled.jacobian)
    workspace.logits .= E.(filled.logits)
    return workspace
end

"""`InPlaceMatrixBLS` whose bordered matrix is assembled in `E`, then factored and solved in the wider `wide`'s type."""
struct WideBLS{E <: BlasFloat, W <: BlasFloat, S <: BifurcationKit.AbstractLinearSolver} <: BifurcationKit.AbstractBorderedLinearSolver
    narrow::Matrix{E}
    wide::Matrix{W}
    pivots::Vector{BlasInt}
    solver::S
end

function (bls::WideBLS{E})(
    jacobian::Any,
    dR::AbstractVector,
    dzu::AbstractVector,
    dzp::T,
    residual::AbstractVector,
    border_value::T,
    ξu::T = one(T),
    ξp::T = one(T);
    dotp::Any = nothing,
    applyξu!::Any = nothing,
)::Tuple where {E, T <: Number}
    GameTracing.bordered_matrix!(bls.narrow, BifurcationKit.getmatrix(jacobian), dR, dzu, dzp, ξu, ξp, applyξu!)
    copyto!(bls.wide, bls.narrow)
    rhs = eltype(bls.wide).(vcat(residual, border_value))
    width = length(rhs) - 1
    if istril(bls.wide) || istriu(bls.wide)
        solution = bls.wide \ rhs
        return E.(view(solution, 1:width)), E(last(solution)), true, 1
    end
    _, _, info = LAPACK.getrf!(bls.wide, bls.pivots; check = true)
    if info > 0
        throw(SingularException(info))
    end
    LAPACK.getrs!('N', bls.wide, bls.pivots, rhs)
    return E.(view(rhs, 1:width)), E(last(rhs)), true, 1
end

function BifurcationKit.bordered_determinant_sign(
    bls::WideBLS,
    iter::BifurcationKit.AbstractContinuationIterable,
    state::BifurcationKit.AbstractContinuationState,
    jacobian::Any,
    dFdl::AbstractVector,
)::eltype(dFdl)
    state_weight = BifurcationKit.getθ(iter)
    GameTracing.bordered_matrix!(
        bls.narrow,
        BifurcationKit.getmatrix(jacobian),
        dFdl,
        state.τ.u,
        state.τ.p,
        state_weight,
        one(state_weight) - state_weight,
        BifurcationKit._get_apply_dot(BifurcationKit.getdot(iter)),
    )
    copyto!(bls.wide, bls.narrow)
    _, _, info = LAPACK.getrf!(bls.wide, bls.pivots; check = false)
    return eltype(dFdl)(last(logabsdet(LU(bls.wide, bls.pivots, info))))
end

"""A `WideBLS` bordering a `width`-row Jacobian in `E`, solved in `widen(E)`, as `matrix_bls` takes `width`."""
function wide_bls(width::Integer, ::Type{E})::WideBLS where {E <: BlasFloat}
    rows = width + 1
    return WideBLS(Matrix{E}(undef, rows, rows), Matrix{widen(E)}(undef, rows, rows), Vector{BlasInt}(undef, rows), BifurcationKit.DefaultLS())
end

"""The unfolded pitchfork's game in arm `arm`'s pass at `precision`: B and D form the branch system in `widen(E)`."""
function stall_game(arm::AbstractString, precision::E)::LogitGame where {E <: Real}
    if arm in ("B", "D")
        return LogitGame(WidePass(NormalFormPass(map(table -> widen(E).(table), UNFOLDED_PITCHFORK_GAME))), precision)
    end
    return LogitGame(NormalFormPass(map(table -> E.(table), UNFOLDED_PITCHFORK_GAME)), precision)
end

"""The unfolded pitchfork traced in `game` under `settings`, a `Function` as `collect_test_logs` takes."""
struct StallTrace{G, S} <: Function
    game::G
    settings::S
end

(trace::StallTrace)()::TraceResult = trace_endpoint(trace.game, length(UNFOLDED_PITCHFORK_GAME), trace.settings)

"""One arm's outcome: end cause, λ reached, steps, corrections near λ 0.808 and the endpoint gap to arm E's `reference`."""
function stall_report(arm::AbstractString, precision::Real, reference::AbstractVector)::String
    E = ifelse(arm == "E", typeof(float(1)), Float32)
    settings = arm_settings(ARMS.fisher_log_precision_reach, E(1) / 100)
    logs, result = collect_test_logs(StallTrace(stall_game(arm, E(precision)), settings); min_level = Debug)
    records = [record for record in logs if record.group == :trace]
    corrections = [
        record.kwargs[:correction] for record in records if record.message == "corrector correction" &&
            first(STALL_WINDOW) <= record.kwargs[:relative_precision] * precision <= last(STALL_WINDOW)
    ]
    stops = [string(get(record.kwargs, :reason, "")) for record in records if record.message == "corrector stopped"]
    tally = join([@sprintf("%s×%d", reason, count(==(reason), stops)) for reason in unique(stops)], " ")
    correction_range = ifelse(isempty(corrections), "none", @sprintf("%.2e..%.2e (n=%d)", minimum(corrections; init = Inf32), maximum(corrections; init = zero(Float32)), length(corrections)))
    if is_error(result)
        context = unwrap_error(result).context
        return @sprintf("STALL arm %s: %s λ=%.6f λmax=%.6f steps=%d corrections %s stops %s gap n/a",
            arm, string(unwrap_error(result).cause), context.precision, context.precision_max, context.steps, correction_range, tally)
    end
    traced = unwrap(result)
    return @sprintf("STALL arm %s: reached λ=%.6f steps=%d accuracy=%.2e corrections %s stops %s gap=%.3e",
        arm, precision, traced.steps, traced.logit_accuracy, correction_range, tally, maximum(abs, traced.logits .- reference))
end

"""Arm E's endpoint logits at `precision`, the reference every arm's gap is measured to."""
function stall_reference(precision::Real)::Vector
    E = typeof(float(1))
    settings = arm_settings(ARMS.fisher_log_precision_reach, E(1) / 100)
    return unwrap(trace_endpoint(stall_game("E", E(precision)), length(UNFOLDED_PITCHFORK_GAME), settings)).logits
end

"""Print each of `arms`' reports at LogitNash's stopping precision past 4, as the unfolded-pitchfork test traces."""
function stall_experiment(arms::Tuple)::Nothing
    profile_status = LogitNash.solve(map(float, UNFOLDED_PITCHFORK_GAME); stop_lambda = float(4), stop_eps = zero(float(4)))
    precision = Float32(last(profile_status).lambda)
    reference = stall_reference(precision)
    for arm in arms
        println(stall_report(arm, precision, reference))
    end
    return nothing
end
