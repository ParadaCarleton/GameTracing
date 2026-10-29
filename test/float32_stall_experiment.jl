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

"""
Arms B1 and B2: the branch system formed from the `E` pass `narrow`, with its residual (B1, `wide_residual`) or its
Jacobian (B2) formed instead from the `widen(E)` pass `wide` and rounded to `E`.
"""
struct SplitPass{P, W}
    narrow::P
    wide::W
    wide_residual::Bool
end

function GameTracing.branch_system!(
    workspace::GameTracing.BranchWorkspace{E},
    game::LogitGame{<:SplitPass},
    point::AbstractVector{E},
    pin::Real,
)::typeof(workspace) where {E <: Real}
    (; narrow, wide, wide_residual) = game.pass!
    narrow_game = LogitGame(narrow, game.target_precision)
    wide_game = LogitGame(wide, widen(E)(game.target_precision))
    wide_point = widen(E).(point)
    narrow_filled = GameTracing.branch_system!(GameTracing.BranchWorkspace(narrow_game, point), narrow_game, point, pin)
    wide_filled = GameTracing.branch_system!(GameTracing.BranchWorkspace(wide_game, wide_point), wide_game, wide_point, widen(E)(pin))
    workspace.residual .= ifelse(wide_residual, E.(wide_filled.residual), narrow_filled.residual)
    workspace.jacobian .= ifelse(wide_residual, narrow_filled.jacobian, E.(wide_filled.jacobian))
    workspace.logits .= narrow_filled.logits
    return workspace
end

"""Arm B3: the `E` pass `narrow`, its residual's gap rows formed as `fma(-P, G, w)`, one rounding in place of two."""
struct FusedPass{P}
    narrow::P
end

function GameTracing.branch_system!(
    workspace::GameTracing.BranchWorkspace{E},
    game::LogitGame{<:FusedPass},
    point::AbstractVector{E},
    pin::Real,
)::typeof(workspace) where {E <: Real}
    (; residual, jacobian, logits) = workspace
    gaps = GameTracing.payoff_gaps(point)
    width = length(gaps)
    relative_precision = point[end]
    logits .= relative_precision .* gaps
    precision = relative_precision * game.target_precision
    target = game.target_precision
    gain = view(residual, 1:width)
    gain_slope = view(jacobian, 1:width, 1:width)
    precision_column = view(jacobian, 1:width, width + 1)
    GameTracing.run_pass!(game.pass!.narrow, nothing, gain, gain_slope, precision_column, logits, precision)
    GameTracing.mul!(precision_column, gain_slope, gaps, -target, -target^2)
    gain_slope .= .-precision .* gain_slope
    for diagonal in axes(gain_slope, 1)
        gain_slope[diagonal, diagonal] += one(E)
    end
    gain .= fma.(-target, gain, gaps)
    fill!(view(jacobian, width + 1, 1:width), zero(E))
    jacobian[end, end] = one(E)
    residual[end] = relative_precision - pin
    return workspace
end

"""Arm B4: a pass that runs the `widen(E)` pass `wide` and rounds its gains and slopes to the trace's `E`."""
struct WideGains{W}
    wide::W
end

function (gains::WideGains)(
    gain::AbstractVector{E},
    gain_slope::AbstractMatrix{E},
    gain_precision_slope::AbstractVector{E},
    logits::AbstractVector{E},
    precision::Real,
)::Nothing where {E <: Real}
    wide_gain, wide_slope, wide_precision_slope = widen(E).(gain), widen(E).(gain_slope), widen(E).(gain_precision_slope)
    gains.wide(wide_gain, wide_slope, wide_precision_slope, widen(E).(logits), widen(E)(precision))
    gain .= E.(wide_gain)
    gain_slope .= E.(wide_slope)
    gain_precision_slope .= E.(wide_precision_slope)
    return nothing
end

"""A pass that runs the pass `narrow` in its payoffs' element type and writes its gains and slopes into wider arrays."""
struct NarrowGains{P}
    narrow::P
end

function (gains::NarrowGains)(
    gain::AbstractVector{W},
    gain_slope::AbstractMatrix{W},
    gain_precision_slope::AbstractVector{W},
    logits::AbstractVector{W},
    precision::Real,
)::Nothing where {W <: Real}
    narrow_type = eltype(first(gains.narrow.payoffs))
    narrow_gain, narrow_slope = narrow_type.(gain), narrow_type.(gain_slope)
    narrow_precision_slope = narrow_type.(gain_precision_slope)
    gains.narrow(narrow_gain, narrow_slope, narrow_precision_slope, narrow_type.(logits), narrow_type(precision))
    gain .= narrow_gain
    gain_slope .= narrow_slope
    gain_precision_slope .= narrow_precision_slope
    return nothing
end

"""Arm B5: the `E` pass `narrow`, its gains and slopes assembled into the branch system in `widen(E)`, then rounded."""
struct WideAssembly{P}
    narrow::P
end

function GameTracing.branch_system!(
    workspace::GameTracing.BranchWorkspace{E},
    game::LogitGame{<:WideAssembly},
    point::AbstractVector{E},
    pin::Real,
)::typeof(workspace) where {E <: Real}
    wide_game = LogitGame(NarrowGains(game.pass!.narrow), widen(E)(game.target_precision))
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
    narrow = NormalFormPass(map(table -> E.(table), UNFOLDED_PITCHFORK_GAME))
    wide = NormalFormPass(map(table -> widen(E).(table), UNFOLDED_PITCHFORK_GAME))
    if arm in ("B", "D")
        return LogitGame(WidePass(wide), precision)
    elseif arm in ("B1", "B2")
        return LogitGame(SplitPass(narrow, wide, arm == "B1"), precision)
    elseif arm == "B3"
        return LogitGame(FusedPass(narrow), precision)
    elseif arm == "B4"
        return LogitGame(WideGains(wide), precision)
    elseif arm == "B5"
        return LogitGame(WideAssembly(narrow), precision)
    end
    return LogitGame(narrow, precision)
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
    for arm in ("B4", "E")
        jump_report(arm, precision, (0.79f0, 0.9f0))
    end
    # noise_report and first_step_endpoint ran in run 40816864095.
    return nothing
end

"""
Print every corrector iterate, stop and accepted point of arm `arm` with λ in `window`: its λ, logits and correction,
or the stop's message and reason.
"""
function noise_report(arm::AbstractString, precision::Real, window::Tuple)::Nothing
    E = ifelse(arm == "E", typeof(float(1)), Float32)
    settings = arm_settings(ARMS.fisher_log_precision_reach, E(1) / 100)
    logs, _ = collect_test_logs(StallTrace(stall_game(arm, E(precision)), settings); min_level = Debug)
    level = precision
    for record in logs
        if record.group != :trace
            continue
        end
        level = get(record.kwargs, :relative_precision, level / precision) * precision
        if !(first(window) <= level <= last(window))
            continue
        end
        if record.message == "corrector correction"
            logits = record.kwargs[:logits]
            println(@sprintf("NOISE %s iterate λ=%.6f z=(%.5f, %.5f) correction=%.3e", arm, level, first(logits), last(logits), record.kwargs[:correction]))
        elseif record.message == "trace point"
            state = record.kwargs[:state]
            logits = last(state) .* state[1:(end - 1)]
            println(@sprintf("NOISE %s accepted λ=%.6f z=(%.5f, %.5f)", arm, level, first(logits), last(logits)))
        elseif record.message in ("corrector stopped", "corrector refused", "trace stopped")
            println("NOISE ", arm, " ", record.message, " ", get(record.kwargs, :reason, ""), " ", get(record.kwargs, :correction, ""))
        end
    end
    return nothing
end

"""The equilibrium arm E reaches at `precision` with its `first_step` scaled by `factor`, and its endpoint logits."""
function first_step_endpoint(precision::Real, factor::Real)::String
    E = typeof(float(1))
    first_step = TraceSettings{E}().first_step * factor
    settings = arm_settings(ARMS.fisher_log_precision_reach, E(1) / 100; first_step)
    result = trace_endpoint(stall_game("E", E(precision)), length(UNFOLDED_PITCHFORK_GAME), settings)
    if is_error(result)
        return string("failed ", unwrap_error(result).cause)
    end
    logits = unwrap(result).logits
    named = ifelse(all(>(5), logits), "(1, 1)", ifelse(all(<(-5), logits), "(2, 2)", "other"))
    return string(named, " ", logits, " steps ", unwrap(result).steps)
end

"""
Print arm `arm`'s accepted points with λ in `window`: λ, logits, step length, predictor move, last correction,
reach move over predictor move, angle and the solve's contraction rates, then its endpoint logits.
"""
function jump_report(arm::AbstractString, precision::Real, window::Tuple)::Nothing
    E = ifelse(arm == "E", typeof(float(1)), Float32)
    settings = arm_settings(ARMS.fisher_log_precision_reach, E(1) / 100)
    logs, result = collect_test_logs(StallTrace(stall_game(arm, E(precision)), settings); min_level = Debug)
    records = [record for record in logs if record.group == :trace]
    # Each solve logs its corrections, then "corrector closed", then "corrector closing" once it converged.
    solve_start, rates, closing, previous_tangent = 1, Float32[], nothing, nothing
    for (position, record) in pairs(records)
        if record.message == "corrector closed"
            solve = view(records, solve_start:position)
            corrections = [entry.kwargs[:correction] for entry in solve if entry.message == "corrector correction"]
            rates = corrections[2:end] ./ corrections[1:(end - 1)]
            solve_start = position + 1
        elseif record.message == "corrector closing"
            closing = (; record.kwargs..., rates)
        elseif record.message == "trace point"
            state = record.kwargs[:state]
            level = last(state) * precision
            # The logged tangent is BifurcationKit's `τ.u` over `[w; ρ]`; its cosine with the last one is plain.
            tangent = record.kwargs[:tangent]
            cosine = ifelse(isnothing(previous_tangent), NaN32,
                sum(tangent .* something(previous_tangent, tangent)) / sqrt(sum(abs2, tangent) * sum(abs2, something(previous_tangent, tangent))))
            if first(window) <= level <= last(window)
                logits = last(state) .* state[1:(end - 1)]
                println(@sprintf("TANGENT %s step=%d λ=%.6f z=(%.5f, %.5f) dρ=%.3e cosine=%.4f closing=%s",
                    arm, record.kwargs[:step], level, first(logits), last(logits), last(tangent), cosine, string(!isnothing(closing))))
                if !isnothing(closing)
                    println(@sprintf("JUMP %s λ=%.6f z=(%.5f, %.5f) ds=%.3e predictor_move=%.3e correction=%.3e reach_ratio=%.3f angle=%.4f rates=%s",
                        arm, level, first(logits), last(logits), closing.step_length, closing.predictor_move, closing.correction,
                        closing.reach_move / closing.predictor_move, closing.angle, join([@sprintf("%.3f", rate) for rate in closing.rates], ",")))
                end
            end
            previous_tangent, closing = tangent, nothing
        end
    end
    if is_error(result)
        println("JUMP ", arm, " endpoint failed ", unwrap_error(result).cause)
    else
        println("JUMP ", arm, " endpoint ", unwrap(result).logits)
    end
    return nothing
end
