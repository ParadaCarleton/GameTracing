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

"""`settled_logits` for the two-action payoff tables `payoffs` (`NormalFormPass`)."""
function settled_endpoint(payoffs::Tuple, logits::AbstractVector{E}, precision::Real, settings::TraceSettings{E})::Bool where {E <: Real}
    return settled_logits(NormalFormPass(map(table -> E.(table), payoffs)), logits, precision, settings)
end

"""
    cross_checked!(payoffs, precision, settings)

Trace `payoffs` to LogitNash's stopping precision past `precision` and test the endpoint against Gambit's branch there,
the oracle, and against `branch_reference`, the Julia cross-check. LogitNash is a cross-check too: a disagreement with it
fails only when Gambit sides with LogitNash. Near an
unfolded pitchfork LogitNash can cross onto the middle arm, as on `UNFOLDED_PITCHFORK_GAME`. A reference that does not
reach the target skips the comparison; every endpoint must also solve the map to the corrector's tolerance.
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
    oracle = gambit_branch(payoffs, cross.precision)
    @test agreeing(traced, oracle.first_choices)
    reference = branch_reference(NormalFormPass(map(float, payoffs)), length(payoffs), cross.precision)
    # A generic game's branch crosses no singular point.
    @test reference.crossings == 0
    if reference.reached
        @test agreeing(traced, reference.first_choices)
    end
    @test settled_endpoint(payoffs, logits, cross.precision, settings)
    if !agreeing(traced, cross.first_choices)
        @test !agreeing(oracle.first_choices, cross.first_choices)
    end
    return nothing
end

@testset "Gambit's λ is the trace's precision: its profile solves z = λ G on the raw payoffs" begin
    rng = fresh_rng()
    for _ in 1:5
        players = rand(rng, 2:3)
        payoffs = ntuple(_ -> 3 .* randn(rng, ntuple(_ -> 2, players)), players)
        precision = 1 + 2 * rand(rng)
        oracle = gambit_branch(payoffs, precision)
        @test abs(oracle.precision - precision) < inv(float(10^6))
        crossing = log.(oracle.first_choices) .- log1p.(-oracle.first_choices)
        response = evaluated(NormalFormPass(payoffs), crossing, oracle.precision).response
        @test maximum(abs, inv.(1 .+ exp.(-response)) .- oracle.first_choices) < inv(float(10^8))
        # Gambit does not normalize payoffs: doubling them doubles λ, and moves the profile.
        doubled = gambit_branch(map(table -> 2 .* table, payoffs), precision / 2)
        @test maximum(abs, doubled.first_choices .- oracle.first_choices) < inv(float(10^8))
        @test maximum(abs, gambit_branch(payoffs, precision / 2).first_choices .- oracle.first_choices) > inv(float(10^4))
    end
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
    for (name, payoffs) in pairs(LOGITNASH_GAMES), precision in (1, 4)
        cross = logitnash_equilibrium(payoffs, precision)
        pass = NormalFormPass(map(float, payoffs))
        reference = branch_reference(pass, length(payoffs), cross.precision)
        @test reference.reached
        # Only the exact battle of the sexes puts a singular point, its pitchfork, on the branch.
        @test reference.crossings <= (name == :battle_of_the_sexes)
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

@testset "past an exact pitchfork the trace ends at an equilibrium or fails as Singular or PrecisionExhausted" begin
    for name in PITCHFORK_GAMES, tangent in (BifurcationKit.Bordered(), BifurcationKit.Secant())
        pass = NormalFormPass(map(table -> Float32.(table), LOGITNASH_GAMES[name]))
        result = trace_endpoint(LogitGame(pass, 4.0f0), length(pass.payoffs), TraceSettings{Float32}(; tangent))
        # The cause depends on whether rounding breaks the symmetry: an exact branch point factors singular, while one
        # that rounding unfolds by about eps (battle_of_the_sexes, rcond ≥ 1e-5) is near-singular and exhausts.
        if is_error(result)
            failure = unwrap_error(result)
            @test failure.cause in (GameTracing.Singular, GameTracing.PrecisionExhausted)
            # A trace that broke at its start never reached the pitchfork.
            @test failure.context.relative_precision > 0
        else
            @test residual_norm(pass, unwrap(result).logits, 4.0f0) < 1.0f-4
        end
    end
end

@testset "a pitchfork unfolded by a payoff perturbation: the bordered trace turns with the principal branch" begin
    # Measured over 40 draws per scale against the arclength reference: the secant tangent followed the principal branch
    # 0/40 at 1e-4, 3/40 at 1e-3, 34/40 at 1e-2 and 39/40 at 1e-1; the bordered tangent 40/40 at every scale.
    rng = fresh_rng()
    for scale in inv.(float.((10^4, 10^3, 10^2, 10))), _ in 1:3
        payoffs = map(table -> table .+ scale .* randn(rng, 2, 2), map(float, LOGITNASH_GAMES.battle_of_the_sexes))
        cross_checked!(payoffs, 4, TraceSettings{Float32}())
    end
end

@testset "a pitchfork unfolded by 1e-4: the reference, the default trace and every arm turn onto the (1, 1) arm" begin
    precision = logitnash_equilibrium(UNFOLDED_PITCHFORK_GAME, 4).precision
    pass = NormalFormPass(map(float, UNFOLDED_PITCHFORK_GAME))
    reference = branch_reference(pass, 2, precision)
    @test reference.reached
    @test reference.crossings == 0
    # Oracle: Gambit 16.7's logit branch and a 50-digit arclength trace both end at the pure equilibrium (1, 1).
    arm_root = level_root(pass, [float(15), float(10)], precision)
    @test all(>(99 // 100), reference.first_choices)
    @test maximum(abs, reference.logits .- arm_root) < inv(float(10^6))
    oracle = gambit_branch(UNFOLDED_PITCHFORK_GAME, precision)
    @test agreeing(reference.first_choices, oracle.first_choices)
    # At Gambit's precision the trace keeps the 1e-4 rule: its accepted corrections all lie within it.
    double_settings = TraceSettings{typeof(float(1))}()
    # A failed double trace throws here, an error of this testset.
    doubled = unwrap(traced_result(UNFOLDED_PITCHFORK_GAME, precision * float(1), double_settings))
    double_logits = doubled.logits
    @test doubled.logit_accuracy <= double_settings.corrector.logit_tolerance
    @test agreeing(first_choices(double_logits), oracle.first_choices)
    @test settled_endpoint(UNFOLDED_PITCHFORK_GAME, double_logits, precision * float(1), double_settings)
    # Float32 may settle at its rounding (`rounding_bound`); it must follow the double-precision branch within that.
    for (label, settings) in ["default" => TraceSettings{Float32}(); arm_variants()]
        result = traced_result(UNFOLDED_PITCHFORK_GAME, precision, settings)
        @test !is_error(result)
        if !is_error(result)
            (; logits, logit_accuracy) = unwrap(result)
            accuracy = max(logit_accuracy, settings.corrector.logit_tolerance)
            @test agreeing(first_choices(logits), oracle.first_choices)
            @test agreeing(first_choices(logits), first_choices(double_logits))
            @test maximum(abs, logits .- double_logits) <= accuracy + doubled.logit_accuracy
            @test newton_correction(NormalFormPass(map(table -> Float32.(table), UNFOLDED_PITCHFORK_GAME)), logits, precision) <= accuracy
            @info "unfolded pitchfork in Float32" label logit_accuracy gap = maximum(abs, logits .- double_logits)
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
        @test agreeing(first_choices(unwrap(result).logits), gambit_branch(OVERSHOT_GAME, precision).first_choices)
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
