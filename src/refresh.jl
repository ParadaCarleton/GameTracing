using LinearAlgebra: Diagonal
using LogExpFunctions: log1pexp, logistic

"""
    choice_kl(from, to)

`KL(Bernoulli(σ(from)) ‖ Bernoulli(σ(to)))` for logits `from` and `to`. Each log ratio is a difference of `log1pexp`
terms weighted by the probability it multiplies, so a large logit costs no digits beyond its own rounding.
"""
function choice_kl(from::E, to::E)::E where {E <: Real}
    return logistic(from) * (log1pexp(-to) - log1pexp(-from)) +
        logistic(-from) * (log1pexp(to) - log1pexp(from))
end

"""
    branch_step(game, logits, relative_precision, state_weight)

One minimum-norm Gauss–Newton step from `x = [asinh(logits); ρ]` onto `game`'s branch `F(x) = 0`, in PALC's norm
`θ ‖du‖² / k + (1 - θ) dρ²` (`θ` the `state_weight`, `k` the number of logits): `dx = -W⁻¹ Jᵀ (J W⁻¹ Jᵀ)⁻¹ F`.
The step moves `ρ` as well, so it stays defined at a fold, where a correction at fixed `ρ` is singular.
"""
function branch_step(
    game::LogitGame,
    logits::AbstractVector{E},
    relative_precision::Real,
    state_weight::Real,
)::Vector{E} where {E <: Real}
    if !(0 < state_weight < 1)
        throw(ArgumentError("state_weight must lie in (0, 1); got $state_weight"))
    end
    width = length(logits)
    system = branch_system(game, [asinh.(logits); E(relative_precision)], relative_precision)
    jacobian = system.jacobian[1:width, :]
    inverse_weight = Diagonal([fill(E(width / state_weight), width); E(1 / (1 - state_weight))])
    weighted = inverse_weight * transpose(jacobian)
    return -(weighted * ((jacobian * weighted) \ system.residual[1:width]))
end

"""The logits at `game`'s branch point nearest `(logits, relative_precision)`, one `branch_step` away."""
function corrected_logits(
    game::LogitGame,
    logits::AbstractVector{E},
    relative_precision::Real,
    state_weight::Real,
)::Vector{E} where {E <: Real}
    return sinh.(asinh.(logits) .+ state_of(branch_step(game, logits, relative_precision, state_weight)))
end

"""
    refresh_kl(current, pending, logits, relative_precision, settings)

How far refreshing a trace's map from `current` to `pending` at the point `(logits, ρ)` moves its choice probabilities:
the largest per-player `choice_kl` from `current`'s to `pending`'s corrected point (`corrected_logits`, in the trace's
norm `settings.state_weight`). Both games must share a target precision, so that `ρ` names one precision in each.
"""
function refresh_kl(
    current::LogitGame,
    pending::LogitGame,
    logits::AbstractVector{E},
    relative_precision::Real,
    settings::TraceSettings,
)::E where {E <: Real}
    if current.target_precision != pending.target_precision
        throw(ArgumentError("a refreshed map keeps its target precision: $(current.target_precision) != $(pending.target_precision)"))
    end
    return maximum(
        choice_kl.(
            corrected_logits(current, logits, relative_precision, settings.state_weight),
            corrected_logits(pending, logits, relative_precision, settings.state_weight),
        ),
    )
end

"""
    refresh_verdict(current, pending, logits, relative_precision, settings, bound)

`(; admitted, kl)`: whether a trace at `(logits, ρ)` may refresh its map from `current` to `pending`, and the `refresh_kl`
that decides it. The refresh is admitted when no player's choice law moves by more than `bound` nats; a non-finite
KL is refused. The caller swaps the map at its next accepted point and owns what a refusal does. The next corrector
must reach the new branch from the old point: an admitted jump is about `sqrt(8 bound)` in logits at even odds, which
should stay inside `settings.corrector.reach` times the step.
"""
function refresh_verdict(
    current::LogitGame,
    pending::LogitGame,
    logits::AbstractVector{E},
    relative_precision::Real,
    settings::TraceSettings,
    bound::Real,
)::NamedTuple{(:admitted, :kl), Tuple{Bool, E}} where {E <: Real}
    kl = refresh_kl(current, pending, logits, relative_precision, settings)
    return (; admitted = kl <= bound, kl)
end
