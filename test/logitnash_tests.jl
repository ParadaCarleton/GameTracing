using BifurcationKit: BifurcationKit
using ErrorTypes: is_error, unwrap, unwrap_error
using LogitNash: LogitNash

"""
LogitNash's QRE of `payoffs` past `precision`: `(; first_choices, precision)`, each player's probability of its first
action and the precision LogitNash stopped at, which may pass the one asked for.
"""
function logitnash_equilibrium(payoffs::Tuple, precision::Real)::NamedTuple
    profile, status = LogitNash.solve(map(float, payoffs); stop_lambda = float(precision), stop_eps = zero(float(precision)))
    return (; first_choices = Float32[first(strategy) for strategy in profile], precision = Float32(status.lambda))
end

"""GameTracing's `TraceResult` for `payoffs` at `precision`."""
function traced_result(payoffs::Tuple, precision::E, settings::TraceSettings{E})::TraceResult{E} where {E <: Real}
    return trace_endpoint(LogitGame(NormalFormPass(map(table -> E.(table), payoffs)), precision), length(payoffs), settings)
end

"""First-action probabilities of `logits`."""
function first_choices(logits::AbstractVector{E})::Vector{E} where {E <: Real}
    return inv.(one(E) .+ exp.(-logits))
end

"""Whether two first-action profiles agree to `1e-3` in every probability."""
agreeing(left::AbstractVector, right::AbstractVector)::Bool = maximum(abs, left .- right) < 1.0f-3

"""
Whether `logits` solve `payoffs` at `precision` to the corrector's tolerance there: at relative precision one the
gap residual the corrector bounds is the logit residual, and twice the tolerance covers recomputing it.
"""
function settled_endpoint(payoffs::Tuple, logits::AbstractVector{E}, precision::Real, settings::TraceSettings{E})::Bool where {E <: Real}
    tolerance = GameTracing.corrector_tolerance(
        settings.corrector.residual_tolerance,
        settings.corrector.tolerance_floor_ulps,
        [logits; one(E)],
    )
    return residual_norm(NormalFormPass(map(table -> E.(table), payoffs)), logits, precision) < 2 * tolerance
end

"""
    cross_checked!(payoffs, precision, settings)

Trace `payoffs` to LogitNash's stopping precision past `precision` and test the endpoint against `branch_reference`.
LogitNash is a cross-check: a disagreement with it fails only when the reference sides with LogitNash. Near an
unfolded pitchfork LogitNash crosses onto the symmetric arm (5 of 400 battle-of-the-sexes games perturbed by 0.05 to
0.1, and most at 1e-4); the reference and the bordered trace did not. A branch that folds back in precision has no
natural-continuation reference; every endpoint must also solve the map to the corrector's tolerance.
"""
function cross_checked!(payoffs::Tuple, precision::Real, settings::TraceSettings{Float32})::Nothing
    cross = logitnash_equilibrium(payoffs, precision)
    result = traced_result(payoffs, cross.precision, settings)
    @test !is_error(result)
    if is_error(result)
        @info "trace failed" payoffs unwrap_error(result).cause
        return nothing
    end
    logits = unwrap(result).logits
    traced = first_choices(logits)
    reference = branch_reference(NormalFormPass(map(float, payoffs)), length(payoffs), cross.precision)
    if reference.reached
        @test agreeing(traced, reference.first_choices)
    end
    @test settled_endpoint(payoffs, logits, cross.precision, settings)
    if !agreeing(traced, cross.first_choices) && reference.reached
        @test !agreeing(reference.first_choices, cross.first_choices)
    end
    return nothing
end

@testset "NormalFormPass: the logit slope is the derivative of the response" begin
    rng = fresh_rng()
    for payoffs in values(LOGITNASH_GAMES)
        pass = NormalFormPass(map(table -> float.(table), payoffs))
        logits = randn(rng, length(payoffs))
        step = inv(float(10^6))
        level = float(3 // 2)
        slope = evaluated(pass, logits, level).logit_slope
        for (rival, _) in pairs(logits)
            nudge = step .* (eachindex(logits) .== rival)
            upper, lower = evaluated(pass, logits .+ nudge, level), evaluated(pass, logits .- nudge, level)
            @test maximum(abs, (upper.response .- lower.response) ./ (2 * step) .- slope[:, rival]) < inv(float(10^6))
        end
        @test evaluated(pass, logits, level).precision_slope ≈ evaluated(pass, logits, 1).response
    end
end

@testset "branch_reference: each endpoint solves the map, and LogitNash's equilibria solve NormalFormPass" begin
    for payoffs in values(LOGITNASH_GAMES), precision in (1, 4)
        cross = logitnash_equilibrium(payoffs, precision)
        pass = NormalFormPass(map(float, payoffs))
        reference = branch_reference(pass, length(payoffs), cross.precision)
        @test reference.reached
        @test residual_norm(pass, reference.logits, cross.precision * float(1)) < inv(float(10^6))
        # In probability space: LogitNash's profile may round a choice to certainty, an infinite logit.
        crossing = log.(cross.first_choices) .- log1p.(-cross.first_choices)
        response = evaluated(NormalFormPass(map(table -> Float32.(table), payoffs)), crossing, cross.precision).response
        @test maximum(abs, inv.(1 .+ exp.(-response)) .- cross.first_choices) < 1.0f-4
    end
end

@testset "the trace reaches the reference equilibrium on LogitNash's games" begin
    for (name, payoffs) in pairs(LOGITNASH_GAMES), precision in (1, 4), tangent in (BifurcationKit.Bordered(), BifurcationKit.Secant())
        if !(name in PITCHFORK_GAMES)
            cross_checked!(payoffs, precision, TraceSettings{Float32}(; tangent))
        end
    end
end

@testset "past an exact pitchfork the trace ends at an equilibrium or fails as Singular" begin
    for name in PITCHFORK_GAMES, tangent in (BifurcationKit.Bordered(), BifurcationKit.Secant())
        pass = NormalFormPass(map(table -> Float32.(table), LOGITNASH_GAMES[name]))
        result = trace_endpoint(LogitGame(pass, 4.0f0), length(pass.payoffs), TraceSettings{Float32}(; tangent))
        # Step halving can bisect onto the branch point itself, where the bordered system is exactly singular.
        if is_error(result)
            failure = unwrap_error(result)
            @test failure.cause == GameTracing.Singular
            # A trace that broke at its start never reached the pitchfork.
            @test failure.context.relative_precision > 0
        else
            @test residual_norm(pass, unwrap(result).logits, 4.0f0) < 1.0f-4
        end
    end
end

@testset "a pitchfork unfolded by a payoff perturbation: the bordered trace turns with the principal branch" begin
    # Measured over 4 draws per scale: the secant tangent followed the principal branch 0/4 at 1e-4, 1/4 at 1e-3, 3/4 at
    # 1e-2 and 4/4 at 1e-1; the bordered tangent 4/4 at every scale.
    rng = fresh_rng()
    for scale in inv.(float.((10^4, 10^3, 10^2, 10))), _ in 1:3
        payoffs = map(table -> table .+ scale .* randn(rng, 2, 2), map(float, LOGITNASH_GAMES.battle_of_the_sexes))
        cross_checked!(payoffs, 4, TraceSettings{Float32}())
    end
end

@testset "a pitchfork unfolded by 1e-4: the default trace and every arm stay on the reference branch" begin
    precision = logitnash_equilibrium(UNFOLDED_PITCHFORK_GAME, 4).precision
    reference = branch_reference(NormalFormPass(map(float, UNFOLDED_PITCHFORK_GAME)), 2, precision)
    @test reference.reached
    for (_, settings) in ["default" => TraceSettings{Float32}(); arm_variants()]
        result = traced_result(UNFOLDED_PITCHFORK_GAME, precision, settings)
        @test !is_error(result)
        if !is_error(result)
            logits = unwrap(result).logits
            # Broken: every setting crosses to the other arm; a fix turns this into an unexpected pass.
            @test_broken agreeing(first_choices(logits), reference.first_choices)
            @test settled_endpoint(UNFOLDED_PITCHFORK_GAME, logits, precision, settings)
        end
    end
end

@testset "a step past the target precision lands where the branch first reaches it" begin
    # Traced at the requested precisions themselves, both short of the fold, so the reference must reach them.
    settings = TraceSettings{Float32}()
    for precision in (3.0f0, 3.1f0)
        result = traced_result(OVERSHOT_GAME, precision, settings)
        @test !is_error(result)
        reference = branch_reference(NormalFormPass(map(float, OVERSHOT_GAME)), length(OVERSHOT_GAME), precision)
        @test reference.reached
        @test agreeing(first_choices(unwrap(result).logits), reference.first_choices)
        @test settled_endpoint(OVERSHOT_GAME, unwrap(result).logits, precision, settings)
    end
end

@testset "random games with two actions each: the trace follows the principal branch" begin
    rng = fresh_rng()
    for _ in 1:20
        players = rand(rng, 2:4)
        payoffs = ntuple(_ -> 3 .* randn(rng, ntuple(_ -> 2, players)), players)
        cross_checked!(payoffs, 3, TraceSettings{Float32}())
    end
end
