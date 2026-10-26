using GameTracing: branch_step, branch_system, choice_kl, corrected_logits
using LinearAlgebra: Diagonal, dot, norm, nullspace
using Random: RandomDevice, Xoshiro

"""`KL(Bernoulli(σ(from)) ‖ Bernoulli(σ(to)))` from its definition, in `BigFloat`."""
function reference_kl(from::Real, to::Real)::BigFloat
    # Each probability and its complement computed directly, so neither rounds to 0 or 1 at a large logit.
    chance, rival = inv(1 + exp(-big(from))), inv(1 + exp(-big(to)))
    rest, rival_rest = inv(1 + exp(big(from))), inv(1 + exp(big(to)))
    return chance * log(chance / rival) + rest * log(rest / rival_rest)
end

"""`log(1 + exp(logit))` in `BigFloat`."""
big_softplus(logit::Real)::BigFloat = log1p(exp(big(logit)))

"""The rounding scale of `choice_kl(from, to)`: each `log1pexp` it evaluates, weighted by the probability multiplying it."""
function kl_rounding_scale(from::Real, to::Real)::BigFloat
    chance = inv(1 + exp(-big(from)))
    return chance * (big_softplus(-from) + big_softplus(-to)) + (1 - chance) * (big_softplus(from) + big_softplus(to))
end

@testset "choice_kl is the Bernoulli KL to within the rounding of its four log1pexp terms" begin
    rng = Xoshiro(rand(RandomDevice(), UInt))
    for _ in 1:20_000
        from = Float32(rand(rng, (1, 10, 40)) * randn(rng))
        to = from + Float32(rand(rng, (1.0f-4, 1.0f-2, 1, 10)) * randn(rng))
        error = abs(choice_kl(from, to) - reference_kl(from, to))
        # An underflowed weight is off by a subnormal spacing, times a log ratio of at most |to - from|.
        underflow = 4 * nextfloat(0.0f0) * (1 + abs(to - from))
        @test error <= 4 * eps(Float32) * kl_rounding_scale(from, to) + underflow
    end
end

@testset "choice_kl: zero at equal laws, symmetric under relabeling the actions" begin
    rng = Xoshiro(rand(RandomDevice(), UInt))
    for _ in 1:200
        from, to = 20 .* randn(rng, Float32, 2)
        @test choice_kl(from, from) == 0
        @test choice_kl(from, to) == choice_kl(-from, -to)
        @test choice_kl(from, to) > -4 * eps(Float32) * kl_rounding_scale(from, to)
    end
end

"""PALC's weights `[θ / k … θ / k, 1 - θ]` over a branch point of `width` logits."""
function palc_weights(width::Integer, state_weight::E)::Diagonal{E} where {E <: Real}
    return Diagonal([fill(state_weight / width, width); 1 - state_weight])
end

"""`count` random points near `logits`: `(; logits, precisions)`, one point per column and relative precision."""
function nearby_points(rng::Xoshiro, logits::AbstractVector{E}, count::Integer)::NamedTuple where {E <: Real}
    return (;
        logits = logits .+ E(1 // 10) .* randn(rng, E, length(logits), count),
        precisions = E(1 // 2) .+ E(2 // 5) .* rand(rng, E, count),
    )
end

@testset "branch_step is the minimum-norm Newton step in PALC's norm" begin
    rng = Xoshiro(rand(RandomDevice(), UInt))
    games = (
        LogitGame(LinearGainPass(float.(GAME.interaction), float.(GAME.stand_alone)), float(TARGET_PRECISION)),
        LogitGame(ThresholdPass(float(4), float(2)), float(1)),
        LogitGame(NormalFormPass(map(table -> float.(table), LOGITNASH_GAMES.three_prisoners)), float(2)),
    )
    for (game, width) in zip(games, (2, 1, 3)), state_weight in (float(1 // 5), float(1 // 2), float(9 // 10))
        points = nearby_points(rng, zeros(width), 5)
        for (logits, relative_precision) in zip(eachcol(points.logits), points.precisions)
            step = branch_step(game, logits, relative_precision, state_weight)
            system = branch_system(game, [asinh.(logits); relative_precision], relative_precision)
            jacobian = system.jacobian[1:width, :]
            residual = system.residual[1:width]
            # Newton's linearized equation holds, and no part of the step runs along the branch's tangent.
            @test norm(jacobian * step .+ residual) <= inv(float(10^10)) * (1 + norm(residual))
            tangent = vec(nullspace(jacobian))
            @test abs(dot(palc_weights(width, state_weight) * step, tangent)) <= inv(float(10^10)) * (1 + norm(step))
            # A Newton step: the residual it leaves is second order in the one it started from.
            moved = [asinh.(logits); relative_precision] .+ step
            @test norm(branch_system(game, moved, last(moved)).residual[1:width]) <= 100 * norm(residual)^2
            @test corrected_logits(game, logits, relative_precision, state_weight) == sinh.(asinh.(logits) .+ step[1:width])
        end
    end
end

@testset "branch_step rejects a state weight outside (0, 1)" begin
    for state_weight in (0.0f0, 1.0f0, -0.5f0, 1.5f0, NaN32)
        @test_throws ArgumentError branch_step(LogitGame(GAME, TARGET_PRECISION), zeros(Float32, WIDTH), 0.5f0, state_weight)
    end
end

@testset "corrected_logits leaves a point on the branch where it is" begin
    traced = trace_endpoint(LogitGame(GAME, TARGET_PRECISION), WIDTH, TraceSettings{Float32}())
    corrected = corrected_logits(LogitGame(GAME, TARGET_PRECISION), traced.logits, 1.0f0, 0.5f0)
    @test maximum(abs, corrected .- traced.logits) < 1.0f-4
end

"""`GAME` in element type `E` with its stand-alone gains moved by `shift` along `direction`."""
function shifted_game(direction::AbstractVector{E}, shift::E)::LogitGame where {E <: Real}
    return LogitGame(LinearGainPass(E.(GAME.interaction), E.(GAME.stand_alone) .+ shift .* direction), E(TARGET_PRECISION))
end

@testset "refresh_kl: zero for the same map, second order in a small change of map" begin
    rng = Xoshiro(rand(RandomDevice(), UInt))
    settings = TraceSettings{Float32}()
    current = shifted_game(zeros(2), zero(float(1)))
    points = nearby_points(rng, [3, -2] ./ 10, 10)
    for (logits, relative_precision) in zip(eachcol(points.logits), points.precisions)
        @test refresh_kl(current, current, logits, relative_precision, settings) == 0
        direction = randn(rng, 2)
        coarse = refresh_kl(current, shifted_game(direction, inv(float(1000))), logits, relative_precision, settings)
        fine = refresh_kl(current, shifted_game(direction, inv(float(2000))), logits, relative_precision, settings)
        @test 7 // 2 < coarse / fine < 9 // 2
    end
end

@testset "refresh_kl is the largest per-player KL from the current map's corrected point to the pending one's" begin
    rng = Xoshiro(rand(RandomDevice(), UInt))
    settings = TraceSettings{Float32}(state_weight = 0.3f0)
    current = LogitGame(GAME, TARGET_PRECISION)
    pending = shifted_game(Float32[0.4, -1], 0.1f0)
    points = nearby_points(rng, Float32[0.3f0, -0.2f0], 10)
    for (logits, relative_precision) in zip(eachcol(points.logits), points.precisions)
        before = corrected_logits(current, logits, relative_precision, 0.3f0)
        after = corrected_logits(pending, logits, relative_precision, 0.3f0)
        @test refresh_kl(current, pending, logits, relative_precision, settings) == maximum(choice_kl.(before, after))
        @test refresh_kl(pending, current, logits, relative_precision, settings) == maximum(choice_kl.(after, before))
    end
end

@testset "refresh_kl rejects maps with different target precisions" begin
    moved = LogitGame(GAME, 2 * TARGET_PRECISION)
    @test_throws ArgumentError refresh_kl(LogitGame(GAME, TARGET_PRECISION), moved, zeros(Float32, WIDTH), 0.5f0, TraceSettings{Float32}())
end

@testset "refresh_verdict admits exactly the refreshes within its bound and refuses a non-finite KL" begin
    settings = TraceSettings{Float32}()
    current = LogitGame(GAME, TARGET_PRECISION)
    pending = shifted_game(Float32[1, 1], 0.05f0)
    logits = Float32[0.3f0, -0.2f0]
    kl = refresh_kl(current, pending, logits, 0.5f0, settings)
    @test kl > 0
    @test refresh_verdict(current, pending, logits, 0.5f0, settings, kl) == (; admitted = true, kl)
    @test refresh_verdict(current, pending, logits, 0.5f0, settings, prevfloat(kl)) == (; admitted = false, kl)
    poisoned = LogitGame(PoisonedPass(GAME, 0.0f0), TARGET_PRECISION)
    verdict = refresh_verdict(current, poisoned, logits, 0.5f0, settings, Inf32)
    @test !verdict.admitted
    @test isnan(verdict.kl)
end

# A caller refreshing its map holds state across callbacks: `keep_going` is the only hook that sees an accepted point.
# `map_indices = [current, pending]` is that state, mutated in place as irv-extremism's redraws are.

"""A pass that forwards to `passes[map_indices[1]]`, the map a refreshing trace currently solves."""
struct RefreshingPass{P}
    passes::Vector{P}
    map_indices::Vector{Int}
end

function (pass::RefreshingPass)(logits::AbstractVector, precision::Real)::NamedTuple
    return pass.passes[first(pass.map_indices)](logits, precision)
end

"""An `accept_corrected` that offers the next map at each converged point below `last_offer`, marking it pending if admitted."""
struct OfferNextMap{R <: RefreshingPass, S <: TraceSettings, E <: Real}
    pass::R
    settings::S
    target_precision::E
    bound::E
    last_offer::E
end

function (offer::OfferNextMap)(logits::AbstractVector, relative_precision::Real, converged::Bool)::Bool
    (; pass, settings, target_precision, bound) = offer
    current = first(pass.map_indices)
    if converged && relative_precision < offer.last_offer && current < length(pass.passes)
        verdict = refresh_verdict(
            LogitGame(pass.passes[current], target_precision),
            LogitGame(pass.passes[current + 1], target_precision),
            logits, relative_precision, settings, bound,
        )
        pass.map_indices[2] = ifelse(verdict.admitted, current + 1, current)
    end
    return true
end

"""A `keep_going` that adopts the pending map at each accepted point."""
struct AdoptPendingMap{R <: RefreshingPass}
    pass::R
end

function (adopt::AdoptPendingMap)(::AbstractVector, ::Real)::Bool
    adopt.pass.map_indices[1] = adopt.pass.map_indices[2]
    return true
end

"""Trace `passes` from the first, offering each next one under `bound`: `(; traced, adopted)`, `adopted` the last map's index."""
function refreshing_trace(passes::AbstractVector, target_precision::E, width::Integer, bound::E)::NamedTuple where {E <: Real}
    pass = RefreshingPass(collect(passes), [1, 1])
    settings = TraceSettings{E}()
    traced = trace_endpoint(
        LogitGame(pass, target_precision), width, settings;
        accept_corrected = OfferNextMap(pass, settings, target_precision, bound, E(9 // 10)),
        keep_going = AdoptPendingMap(pass),
    )
    return (; traced, adopted = first(pass.map_indices))
end

"""
`count` copies of `payoffs` with noise of scale `amplitude`, shrinking toward `amplitude / count` in the last: the maps a
sampled game refreshes through. The noise never vanishes, so no map keeps an exact symmetry and its exact pitchfork.
"""
function noisy_tables(rng::Xoshiro, payoffs::Tuple, count::Integer, amplitude::Real)::Vector
    return [map(table -> Float32.(table .+ amplitude * (count + 1 - step) / count .* randn(rng, size(table))), payoffs) for step in 1:count]
end

# Games near a pitchfork are left out: there the arm a refreshed trace keeps is the one its maps chose at the turn,
# not its last map's, and a refresh beside the near-singular turn can strand the corrector (TraceStuck).
@testset "a trace that refreshes its map ends at the equilibrium a plain trace of its last map reaches" begin
    rng = Xoshiro(rand(RandomDevice(), UInt))
    for (name, payoffs) in pairs(LOGITNASH_GAMES)
        if name in PITCHFORK_GAMES
            continue
        end
        tables = noisy_tables(rng, payoffs, 8, 1 // 50)
        # Each admitted jump, about sqrt(8 bound) in logits, sits well inside the corrector's reach of a step.
        refreshed = refreshing_trace(NormalFormPass.(tables), 3.0f0, length(payoffs), 1.0f-4)
        @test refreshed.traced isa Traced
        @test refreshed.adopted > 1
        exact = trace_endpoint(LogitGame(NormalFormPass(tables[refreshed.adopted]), 3.0f0), length(payoffs), TraceSettings{Float32}())
        @test exact isa Traced
        if refreshed.traced isa Traced && exact isa Traced
            @test maximum(abs, refreshed.traced.logits .- exact.logits) < 1.0f-3
        end
    end
end

@testset "a refused refresh leaves the trace exactly as it is without one" begin
    rng = Xoshiro(rand(RandomDevice(), UInt))
    for payoffs in values(LOGITNASH_GAMES)
        passes = NormalFormPass.(noisy_tables(rng, payoffs, 4, 1))
        plain = trace_endpoint(LogitGame(first(passes), 3.0f0), length(payoffs), TraceSettings{Float32}())
        refused = refreshing_trace(passes, 3.0f0, length(payoffs), 0.0f0)
        @test refused.adopted == 1
        @test refused.traced isa Traced
        @test plain isa Traced
        if refused.traced isa Traced && plain isa Traced
            @test refused.traced.logits == plain.logits
        end
    end
end
