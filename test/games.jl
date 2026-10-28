using LinearAlgebra: det, dot, norm, opnorm
using Random: AbstractRNG, RandomDevice, Xoshiro

"""A generator on a fresh seed, logged so a failing draw can be replayed with `Xoshiro(seed)`."""
function fresh_rng()::Xoshiro
    seed = rand(RandomDevice(), UInt)
    @info "fresh draw" seed
    return Xoshiro(seed)
end

"""Payoff gains `G = A p + b` with `p = logistic(logits)`, independent of the precision: the logit map `λ (A p + b)`."""
struct LinearGainPass{E <: Real}
    interaction::Matrix{E}
    stand_alone::Vector{E}
end

function (pass::LinearGainPass)(
    gain::AbstractVector,
    gain_slope::AbstractMatrix,
    gain_precision_slope::AbstractVector,
    logits::AbstractVector{E},
    ::Real,
)::Nothing where {E <: Real}
    stays = inv.(one(E) .+ exp.(-logits))
    gain .= pass.interaction * stays .+ pass.stand_alone
    gain_slope .= pass.interaction .* transpose(stays .* (one(E) .- stays))
    fill!(gain_precision_slope, zero(E))
    return nothing
end

GameTracing.gain_scale(pass::LinearGainPass{E}, ::Real)::E where {E} =
    maximum(sum(abs, pass.interaction; dims = 2)) + maximum(abs, pass.stand_alone)

"""One logit that gains `amplitude * logistic(logits - threshold)`: a coordination game whose branch folds back when `threshold` is large."""
struct ThresholdPass{E <: Real}
    amplitude::E
    threshold::E
end

function (pass::ThresholdPass)(
    gain::AbstractVector,
    gain_slope::AbstractMatrix,
    gain_precision_slope::AbstractVector,
    logits::AbstractVector{E},
    ::Real,
)::Nothing where {E <: Real}
    stays = inv.(one(E) .+ exp.(pass.threshold .- logits))
    gain .= pass.amplitude .* stays
    gain_slope .= reshape(pass.amplitude .* stays .* (one(E) .- stays), 1, 1)
    fill!(gain_precision_slope, zero(E))
    return nothing
end

GameTracing.gain_scale(pass::ThresholdPass{E}, ::Real)::E where {E} = abs(pass.amplitude)

"""A `LinearGainPass` whose gains shrink with the precision: `G = (A p + b) / sqrt(1 + (λ s)²)`, so `∂G/∂λ ≠ 0`."""
struct DampedGainPass{E <: Real}
    pass::LinearGainPass{E}
    spread::E
end

function (pass::DampedGainPass)(
    gain::AbstractVector,
    gain_slope::AbstractMatrix,
    gain_precision_slope::AbstractVector,
    logits::AbstractVector{E},
    precision::Real,
)::Nothing where {E <: Real}
    pass.pass(gain, gain_slope, gain_precision_slope, logits, precision)
    damping = inv(sqrt(1 + (precision * pass.spread)^2))
    gain_precision_slope .= -precision * pass.spread^2 * damping^2 .* damping .* gain
    gain .*= damping
    gain_slope .*= damping
    return nothing
end

# The damping is at most one.
GameTracing.gain_scale(pass::DampedGainPass{E}, target_precision::Real)::E where {E} = gain_scale(pass.pass, target_precision)

"""One logit whose gain is the precision itself, `G = λ`: the branch `z = λ²`."""
struct PrecisionGainPass end

function (::PrecisionGainPass)(
    gain::AbstractVector,
    gain_slope::AbstractMatrix,
    gain_precision_slope::AbstractVector,
    ::AbstractVector{E},
    precision::Real,
)::Nothing where {E <: Real}
    fill!(gain, E(precision))
    fill!(gain_slope, zero(E))
    fill!(gain_precision_slope, one(E))
    return nothing
end

GameTracing.gain_scale(::PrecisionGainPass, target_precision::T)::T where {T <: Real} = target_precision

"""A pass whose gain and slopes are NaN once the precision passes `poison`, as a game that overflows there is."""
struct PoisonedPass{P, E <: Real}
    pass::P
    poison::E
end

function (pass::PoisonedPass)(
    gain::AbstractVector,
    gain_slope::AbstractMatrix,
    gain_precision_slope::AbstractVector,
    logits::AbstractVector,
    precision::Real,
)::Nothing
    pass.pass(gain, gain_slope, gain_precision_slope, logits, precision)
    if precision > pass.poison
        fill!(gain, NaN32)
        fill!(gain_slope, NaN32)
        fill!(gain_precision_slope, NaN32)
    end
    return nothing
end

GameTracing.gain_scale(pass::PoisonedPass, target_precision::T)::T where {T <: Real} = gain_scale(pass.pass, target_precision)

"""A pass that gives a NaN slope, so the Jacobian is non-finite where the residual is not."""
struct NaNSlopePass{P}
    pass::P
end

function (pass::NaNSlopePass)(
    gain::AbstractVector,
    gain_slope::AbstractMatrix,
    gain_precision_slope::AbstractVector,
    logits::AbstractVector,
    precision::Real,
)::Nothing
    pass.pass(gain, gain_slope, gain_precision_slope, logits, precision)
    fill!(gain_slope, NaN32)
    return nothing
end

GameTracing.gain_scale(pass::NaNSlopePass, target_precision::T)::T where {T <: Real} = gain_scale(pass.pass, target_precision)

"""A pass that throws a caller's `ArgumentError` once the precision is positive."""
struct ThrowingPass{P}
    pass::P
end

function (pass::ThrowingPass)(
    gain::AbstractVector,
    gain_slope::AbstractMatrix,
    gain_precision_slope::AbstractVector,
    logits::AbstractVector,
    precision::Real,
)::Nothing
    if precision > 0
        throw(ArgumentError("bad game"))
    end
    pass.pass(gain, gain_slope, gain_precision_slope, logits, precision)
    return nothing
end

GameTracing.gain_scale(pass::ThrowingPass, target_precision::T)::T where {T <: Real} = gain_scale(pass.pass, target_precision)

"""
The logit map `T = λ G` of `pass` at `logits` and precision `λ`, in fresh arrays: `(; response, logit_slope,
precision_slope)`, which are `T`, `∂T/∂z = λ ∂G/∂z` and `∂T/∂λ = G + λ ∂G/∂λ`.
"""
function evaluated(pass::Any, logits::AbstractVector{E}, precision::Real)::NamedTuple where {E <: Real}
    width = length(logits)
    gain = Vector{E}(undef, width)
    gain_slope = Matrix{E}(undef, width, width)
    gain_precision_slope = Vector{E}(undef, width)
    pass(gain, gain_slope, gain_precision_slope, logits, precision)
    return (;
        response = precision .* gain,
        logit_slope = precision .* gain_slope,
        precision_slope = gain .+ precision .* gain_precision_slope,
    )
end

"""The largest `|logits - T(logits, precision)|` entry: zero at an equilibrium."""
residual_norm(pass::Any, logits::AbstractVector, precision::Real) = maximum(
    abs,
    logits .- evaluated(pass, logits, precision).response,
)

"""Picard iterates of `logits -> T(logits, precision)` from the flat start: the equilibrium when `T` contracts."""
function picard_equilibrium(
    pass::Any,
    width::Integer,
    precision::E,
    rounds::Integer,
)::Vector{E} where {E <: Real}
    logits = zeros(E, width)
    for _ in 1:rounds
        logits = evaluated(pass, logits, precision).response
    end
    return logits
end

"""
Payoff gains of a normal-form game in which every player has two actions: `G_p = U_p(1, π) - U_p(2, π)`,
each rival `q` playing its first action with probability `logistic(z_q)`. `payoffs[p][profile]` is player `p`'s payoff.
"""
struct NormalFormPass{N, A <: AbstractArray{<:Real, N}}
    payoffs::NTuple{N, A}
end

GameTracing.gain_scale(pass::NormalFormPass{N, A}, ::Real)::eltype(A) where {N, A} =
    maximum(table -> maximum(abs, table), pass.payoffs)

"""The probability that a player whose first action has probability `first_choice` plays `action`."""
function action_probability(first_choice::E, action::Integer)::E where {E <: Real}
    return ifelse(action == 1, first_choice, 1 - first_choice)
end

"""`+1` for a player's first action, `-1` for its second: the sign it carries in a payoff gain."""
action_sign(action::Integer)::Int = 3 - 2 * action

function (pass::NormalFormPass{N})(
    gain::AbstractVector,
    gain_slope::AbstractMatrix,
    gain_precision_slope::AbstractVector,
    logits::AbstractVector{E},
    ::Real,
)::Nothing where {N, E <: Real}
    first_choices = inv.(one(E) .+ exp.(-logits))
    fill!(gain, zero(E))
    fill!(gain_slope, zero(E))
    fill!(gain_precision_slope, zero(E))
    for profile in CartesianIndices(first(pass.payoffs)), (player, payoff) in pairs(pass.payoffs)
        signed = action_sign(profile[player]) * payoff[profile]
        chances = [action_probability(first_choices[rival], profile[rival]) for rival in 1:N]
        gain[player] += signed * prod((chances[rival] for rival in 1:N if rival != player); init = one(E))
        for rival in 1:N
            if rival != player
                others = prod((chances[other] for other in 1:N if other != player && other != rival); init = one(E))
                gain_slope[player, rival] += signed * action_sign(profile[rival]) * others
            end
        end
    end
    gain_slope .*= transpose(first_choices .* (one(E) .- first_choices))
    return nothing
end

"""Whether `logits` solves `pass` at `precision` to `Float64` rounding at `scale`, times the logits' size."""
function settled_root(pass::Any, logits::AbstractVector{E}, precision::E, scale::E)::Bool where {E <: Real}
    return residual_norm(pass, logits, precision) <= 10^4 * eps(scale) * (1 + maximum(abs, logits))
end

"""
    level_root(pass, logits, precision)

The root of `pass` at fixed `precision` that `Float64` Newton reaches from `logits`; an `ErrorException` when 50 steps do not settle.
"""
function level_root(pass::Any, logits::AbstractVector, precision::Real)::AbstractVector
    root, level = float.(collect(logits)) .* float(1), precision * float(1)
    for _ in 1:50
        if settled_root(pass, root, level, max(level, one(level)))
            return root
        end
        gains = evaluated(pass, root, level)
        root = root .- (one(gains.logit_slope) .- gains.logit_slope) \ (root .- gains.response)
    end
    error("level_root: Newton did not settle at precision $level")
end

"""`(; residual, jacobian)` of the branch equations `z - T(z, λ) = 0` of `pass` at `point = [z; λ]`."""
function reference_system(pass::Any, point::AbstractVector{E})::NamedTuple where {E <: Real}
    logits, level = point[1:(end - 1)], last(point)
    gains = evaluated(pass, logits, level)
    return (; residual = logits .- gains.response, jacobian = [one(gains.logit_slope) .- gains.logit_slope -gains.precision_slope])
end

"""The unit tangent of the branch of `pass` at `point`, oriented along `previous`."""
function reference_tangent(pass::Any, point::AbstractVector{E}, previous::AbstractVector{E})::Vector{E} where {E <: Real}
    jacobian = reference_system(pass, point).jacobian
    tangent = [jacobian; transpose(previous)] \ [zeros(E, size(jacobian, 1)); one(E)]
    return tangent ./ norm(tangent)
end

"""The sign of `det(∂F/∂z)` times that of `dλ/ds`: constant along a smooth branch, through folds included."""
function reference_orientation(pass::Any, point::AbstractVector{E}, tangent::AbstractVector{E})::E where {E <: Real}
    jacobian = reference_system(pass, point).jacobian
    return sign(det(jacobian[:, 1:(end - 1)])) * sign(last(tangent))
end

"""
`(; settled, point)`: Newton on the branch equations and the arclength condition `τ ⋅ (y - point) = step`, from the
predictor `point + step τ`; `settled` when it solves the map to `settled_root` at `scale` within `step / 10` of the predictor.
"""
function corrected_point(
    pass::Any,
    point::AbstractVector{E},
    tangent::AbstractVector{E},
    step::E,
    scale::E,
)::NamedTuple where {E <: Real}
    predicted = point .+ step .* tangent
    candidate = copy(predicted)
    for _ in 1:6
        system = reference_system(pass, candidate)
        candidate = candidate .- [system.jacobian; transpose(tangent)] \ [system.residual; dot(tangent, candidate .- point) - step]
        if settled_root(pass, candidate[1:(end - 1)], last(candidate), scale)
            return (; settled = norm(candidate .- predicted) <= step / 10, point = candidate)
        end
    end
    return (; settled = false, point = candidate)
end

"""
    branch_reference(pass, width, precision; start, start_precision)

`(; reached, logits, first_choices, crossings)`: an independent reference for the branch of `pass` at `precision`, by pseudo-arclength
continuation in `Float64` over `[z; λ]` from the root `level_root` reaches from `start` at `start_precision` (default:
zero logits at zero precision, the principal branch). It ends where λ first reaches `precision`, through folds included.

A step of arclength at most `1/100` halves until its corrector settles near the predictor, the tangent turns under
`acos(0.99)`, and `reference_orientation` keeps its sign. A sign change marks a step onto another sheet: on
`UNFOLDED_PITCHFORK_GAME` the principal branch turns steeply past the middle arm, which a step in λ alone lands on.
A sign change that persists to a step of `1e-6` is taken as a crossing of a singular point, as at an exact pitchfork;
`crossings` counts them, and a generic game's branch has none. `reached = false` when the step underflows or `10^6`
steps do not reach `precision`.
"""
function branch_reference(
    pass::Any,
    width::Integer,
    precision::Real;
    start::AbstractVector = zeros(width),
    start_precision::Real = 0,
)::NamedTuple
    # `float(1)` widens a 32-bit precision, which no Float64 literal names.
    target = precision * float(1)
    point = [level_root(pass, start, start_precision); start_precision * float(1)]
    tangent = reference_tangent(pass, point, [zeros(width); one(target)])
    side = reference_orientation(pass, point, tangent)
    step, largest, singular_step = inv(float(1000)), inv(float(100)), inv(float(10^6))
    crossings = 0
    for _ in 1:(10^6)
        if !(last(point) < target) || !(step > eps(target))
            break
        end
        corrected = corrected_point(pass, point, tangent, step, target)
        if corrected.settled
            next_tangent = reference_tangent(pass, corrected.point, tangent)
            next_side = reference_orientation(pass, corrected.point, next_tangent)
            if dot(next_tangent, tangent) >= 99 // 100 && (next_side == side || step <= singular_step)
                crossings += next_side != side
                if !(last(corrected.point) < target)
                    # The step brackets the target: Newton at the target from the bracket's interpolation.
                    weight = (target - last(point)) / (last(corrected.point) - last(point))
                    logits = level_root(pass, point[1:width] .+ weight .* (corrected.point[1:width] .- point[1:width]), target)
                    return (; reached = true, logits, first_choices = inv.(1 .+ exp.(-logits)), crossings)
                end
                point, tangent, side, step = corrected.point, next_tangent, next_side, min(2 * step, largest)
                continue
            end
        end
        step /= 2
    end
    logits = point[1:width]
    return (; reached = last(point) == target, logits, first_choices = inv.(1 .+ exp.(-logits)), crossings)
end

"""
    MapSwap(before, after, precision, swap_level, width)

A trace over `width` logits to `precision` whose game changes from `before` to `after` once the trace reaches relative
precision `swap_level`: a redraw of the map at an accepted point.
"""
struct MapSwap{B, A, E <: Real}
    before::B
    after::A
    precision::E
    swap_level::E
    width::Int
end

# The swap moves a two-player branch by under a tenth: a redraw a fixed-precision correction absorbs.
const NEAR_SWAP = MapSwap(
    LinearGainPass([0 0.8f0; 0.6f0 0], [0.3f0, -0.2f0]),
    LinearGainPass([0 0.8f0; 0.6f0 0], [0.34f0, -0.23f0]),
    4.0f0, 0.5f0, 2,
)

# The swap removes the low branch at precision 1/2: the only root there is the high one, about 3.5 logits away,
# and the homotopy between the maps reaches it with `s` rising throughout (`rising_homotopy`).
const FAR_SWAP = MapSwap(ThresholdPass(8.0f0, 4.0f0), ThresholdPass(8.0f0, 1.5f0), 1.0f0, 0.5f0, 1)

# The new map's root at precision 1/2 lies past a fold of the homotopy: the blended map loses the low root near s = 0.44.
const FOLD_SWAP = MapSwap(ThresholdPass(8.0f0, 4.0f0), ThresholdPass(16.0f0, 2.0f0), 1.0f0, 0.5f0, 1)

# Two symmetric anti-coordination maps at precision 2: the blend's symmetric root keeps its symmetric mode positive
# while its antisymmetric mode changes sign, a pitchfork with no fold (`symmetric_modes`).
const PITCHFORK_SWAP = MapSwap(
    LinearGainPass([0 -2.0f0; -2.0f0 0], [0.0f0, 0.0f0]),
    LinearGainPass([0 -8.0f0; -8.0f0 0], [0.0f0, 0.0f0]),
    4.0f0, 0.5f0, 2,
)

"""
    symmetric_modes(interaction, precision)

`(; symmetric, antisymmetric)`: the two eigenvalues of `I - λ ∂G/∂z` at the symmetric root of `z = λ a σ(z)` on both
logits of a two-player map with off-diagonal interaction `a`, found by Float64 Newton from zero: `1 ∓ λ a σ'(z)`.
"""
function symmetric_modes(interaction::Real, precision::Real)::NamedTuple
    pass = LinearGainPass([0 interaction; interaction 0] .* float(1), zeros(2))
    root = level_root(pass, zeros(2), precision)
    slope = inv(1 + exp(-first(root))) * inv(1 + exp(first(root)))
    return (; symmetric = 1 - precision * interaction * slope, antisymmetric = 1 + precision * interaction * slope)
end

"""`pass`'s gain at the logit `logit`, in Float64."""
threshold_gain(pass::ThresholdPass, logit::Real)::typeof(logit * float(1)) = pass.amplitude * float(1) / (1 + exp(pass.threshold * float(1) - logit))

"""
    rising_homotopy(before, after, precision, logit)

Whether the homotopy `z = λ ((1 - s) G₀(z) + s G₁(z))` from `before`'s root `logit` at precision `λ` reaches `s = 1`
with `s` rising: solved for `s`, the path is `s(z) = (z / λ - G₀) / (G₁ - G₀)`, here sampled upward in `z` from
`logit` in Float64 steps of 1e-4. A fall in `s` is a fold. Each map is one logit, with `G₁ > G₀` on the path.
"""
function rising_homotopy(before::ThresholdPass, after::ThresholdPass, precision::Real, logit::Real)::Bool
    # `float(1)` widens to Float64; the path starts at `s ≈ 0`, so any first sample rises.
    level, blend, logit_step = precision * float(1), typemin(float(1)), inv(float(10^4))
    for logit_now in range(logit * float(1); step = logit_step, length = 10^6)
        spread = threshold_gain(after, logit_now) - threshold_gain(before, logit_now)
        if !(spread > 0)
            throw(ArgumentError("the gains do not order on the path at logit $logit_now"))
        end
        next = (logit_now / level - threshold_gain(before, logit_now)) / spread
        if next >= 1
            return true
        end
        if next < blend
            return false
        end
        blend = next
    end
    throw(ArgumentError("the path does not reach s = 1 within 100 logits"))
end

"""
    SwapAt(game, swap_level)

A `redraw` hook that returns `game` at every accepted point from relative precision `swap_level` on, and `nothing`
before: the trace swaps once, since returning its current game swaps nothing.
"""
struct SwapAt{G, E <: Real}
    game::G
    swap_level::E
end

function (hook::SwapAt{G})(::AbstractVector, relative_precision::Real)::Union{Nothing, G} where {G}
    if relative_precision >= hook.swap_level
        return hook.game
    end
    return nothing
end

"""A `keep_going` hook that stops the trace at its first accepted point at or past `swap_level`."""
struct StopAt{E <: Real}
    swap_level::E
end

(hook::StopAt)(::AbstractVector, relative_precision::Real)::Bool = relative_precision < hook.swap_level

"""`table` in Float32."""
function single_precision(table::AbstractArray{T, N})::Array{Float32, N} where {T <: Real, N}
    return Float32.(table)
end

"""Draws one random payoff table of scale 3 from `rng` for `players` players with two actions each."""
struct RandomTable{R <: AbstractRNG}
    rng::R
    players::Int
end

(table::RandomTable)(::Integer)::Array = 3 .* randn(table.rng, ntuple(Returns(2), table.players))

"""Moves a payoff table by Gaussian noise of scale 1/20 drawn from `rng`, as a redraw moves a sampled map."""
struct PayoffJitter{R <: AbstractRNG}
    rng::R
end

(jitter::PayoffJitter)(table::AbstractArray)::Array = table .+ randn(jitter.rng, size(table)...) ./ 20

"""`players` random payoff tables of scale 3 from `rng`."""
random_payoffs(rng::AbstractRNG, players::Integer)::Tuple = ntuple(RandomTable(rng, players), players)

"""
    random_swap(seed)

`(; swap, reference_before, reference_after)`: a random 3-player game from seed `seed` whose payoffs move by about 0.05
at relative precision 1/2, as a mid-trace redraw moves a sampled map, and its two maps in Float64.
"""
function random_swap(seed::Integer)::NamedTuple
    rng = Xoshiro(1000 + seed)
    before = random_payoffs(rng, 3)
    after = map(PayoffJitter(rng), before)
    swap = MapSwap(NormalFormPass(map(single_precision, before)), NormalFormPass(map(single_precision, after)), 3.0f0, 0.5f0, 3)
    return (; swap, reference_before = NormalFormPass(before), reference_after = NormalFormPass(after))
end

"""
    probability_bound(pass, logits, precision, residual, slope)

The largest probability error of a root of `pass` at `precision` whose logit residual is `residual`, to first order:
the logit error `‖J⁻¹‖∞ residual`, `J = I - ∂(λG)/∂z` at the root `logits`, plus Float32's 32-ulp rounding of the
logits, times `slope`, the probabilities' largest sup-norm slope in the logits: 1/4 for one logistic logit, 1/2 for a
softmax over several.
"""
function probability_bound(pass::Any, logits::AbstractVector, precision::Real, residual::Real, slope::Real)::typeof(float(residual))
    jacobian = reference_system(pass, [logits; precision * float(1)]).jacobian[:, 1:(end - 1)]
    logit_error = opnorm(inv(jacobian), Inf32) * residual + 32 * eps(Float32) * maximum(abs, logits)
    return slope * logit_error
end
