using BifurcationKit: BifurcationKit
using ErrorTypes: is_error, unwrap
using LinearAlgebra: Diagonal, dot

@testset "bernoulli_kl is the Bernoulli KL, zero only at equal logits, finite where probabilities round" begin
    rng = fresh_rng()
    for _ in 1:20
        from, to = 4 .* randn(rng, Float32, 2)
        probability, other = inv(1 + exp(-from)), inv(1 + exp(-to))
        direct = probability * log(probability / other) + (1 - probability) * log((1 - probability) / (1 - other))
        @test GameTracing.bernoulli_kl(from, to) ≈ direct rtol = 1.0f-3 atol = 1.0f-6
        @test GameTracing.bernoulli_kl(from, to) >= 0
    end
    @test GameTracing.bernoulli_kl(0.3f0, 0.3f0) == 0
    @test isfinite(GameTracing.bernoulli_kl(40.0f0, -40.0f0))
end

@testset "chord_angle measures the turn from the predictor, and an empty chord has none" begin
    geometry = TraceGeometry{Float32}(distance = SummedDistance, state_weight = 0.5f0)
    start = zeros(Float32, 3)
    @test GameTracing.chord_angle(start, Float32[1, 0, 0], Float32[2, 0, 0], geometry) ≈ 0 atol = 1.0f-3
    @test GameTracing.chord_angle(start, Float32[1, 0, 0], Float32[0, 1, 0], geometry) ≈ Float32(π / 2)
    @test GameTracing.chord_angle(start, Float32[1, 0, 0], start, geometry) == 0
end

@testset "each distance weighs coordinates and precision as documented, Fisher measuring in the summed norm" begin
    displacement = Float32[1, 2, 3]
    for (distance, expected) in ((AveragedDistance, 0.4f0 * 5 / 2 + 0.6f0 * 9), (SummedDistance, 0.4f0 * 5 + 9), (FisherDistance, 0.4f0 * 5 + 9))
        geometry = TraceGeometry{Float32}(; distance, state_weight = 0.4f0)
        @test GameTracing.arclength_distance(zeros(Float32, 3), displacement, geometry)^2 ≈ expected
    end
end

@testset "PALC's norm in each distance is the trace's own" begin
    point = Float32[0.4f0, -1.2f0, 0.3f0]
    displacement = Float32[0.2f0, -0.1f0, 0.05f0]
    for distance in instances(StepDistance), coordinates in instances(TraceCoordinates)
        geometry = TraceGeometry{Float32}(; coordinates, distance, state_weight = 0.3f0)
        palc = GameTracing.palc_geometry(geometry, 2)
        GameTracing.BifurcationKit.reweight!(palc.dot, GameTracing.BifurcationKit.BorderedArray(point, last(point)))
        # PALC's pin moves with the relative precision, so it repeats the displacement's last coordinate.
        palc_square = palc.dot(displacement, displacement, last(displacement), last(displacement), palc.theta)
        logits = GameTracing.trace_logits(coordinates, point[1:2], last(point))
        weights = ifelse(distance == FisherDistance, inv.(1 .+ exp.(-logits)) .* inv.(1 .+ exp.(logits)), ones(Float32, 2))
        plain = TraceGeometry{Float32}(; coordinates, distance = ifelse(distance == FisherDistance, SummedDistance, distance), state_weight = 0.3f0)
        weighted = [sqrt.(weights) .* displacement[1:2]; last(displacement)]
        @test palc_square ≈ GameTracing.arclength_distance(zeros(Float32, 3), weighted, plain)^2 rtol = 1.0f-5
    end
end

@testset "a converged corrector past the KL or angle limit is refused" begin
    predicted = Float32[1, 0, 0.5f0]
    for (corrector, corrected) in (
            (CorrectorSettings{Float32}(reach = nothing, max_kl = 1.0f-3), Float32[3, 0, 0.5f0]),
            (CorrectorSettings{Float32}(reach = nothing, max_angle = 0.1f0), Float32[0, 1, 0.5f0]),
            (CorrectorSettings{Float32}(reach = 0.5f0), Float32[3, 0, 0.5f0]),
        )
        check = GameTracing.CorrectorReach(
            GameTracing.AcceptAll(),
            corrector,
            TraceGeometry{Float32}(coordinates = Logits, distance = SummedDistance),
            Float32[0, 0, 0.5f0],
            predicted,
            1.0f0,
        )
        @test GameTracing.closing_accepted(check, predicted)
        @test !GameTracing.closing_accepted(check, corrected)
        @test check((; x = corrected, p = 0.5f0, step = 2, residual = 0.0f0); fromNewton = false) === false
    end
end

@testset "a projection in payoff gaps lands on the branch like one in asinh logits" begin
    game = LogitGame(GAME, TARGET_PRECISION)
    traced = unwrap(trace_endpoint(game, WIDTH, TraceSettings{Float32}()))
    for coordinates in instances(TraceCoordinates)
        settings = TraceSettings{Float32}(geometry = TraceGeometry{Float32}(; coordinates, distance = SummedDistance))
        projected = branch_projection(game, traced.logits .+ 1.0f-2, 1.0f0, settings)
        @test residual_norm(GAME, projected.logits, TARGET_PRECISION * projected.relative_precision) < 1.0f-3
    end
end

"""Whether `settings` traces `payoffs` to the fine Float64 reference at `precision`, or to an equilibrium where the branch folds back."""
function reaches_reference(payoffs::Tuple, precision::Real, settings::TraceSettings{Float32})::Bool
    result = traced_result(payoffs, Float32(precision), settings)
    if is_error(result)
        return false
    end
    logits = unwrap(result).logits
    reference = branch_reference(NormalFormPass(map(float, payoffs)), length(payoffs), precision)
    if reference.reached
        return agreeing(first_choices(logits), reference.first_choices)
    end
    return residual_norm(NormalFormPass(map(table -> Float32.(table), payoffs)), logits, Float32(precision)) < 1.0f-3
end

@testset "every arm reaches the reference equilibrium on LogitNash's games, the overshot game and random games" begin
    rng = fresh_rng()
    random_games = [ntuple(_ -> 3 .* randn(rng, ntuple(_ -> 2, players)), players) for players in rand(rng, 2:4, 6)]
    for (label, settings) in arm_variants()
        for (name, payoffs) in pairs(LOGITNASH_GAMES)
            if !(name in PITCHFORK_GAMES)
                @test reaches_reference(payoffs, 4, settings)
            end
        end
        for precision in (3, 31 // 10)
            @test reaches_reference(OVERSHOT_GAME, precision, settings)
        end
        for payoffs in random_games
            reached = reaches_reference(payoffs, 3, settings)
            @test reached
            if !reached
                @info "arm missed the reference" label payoffs
            end
        end
    end
end

@testset "every arm crosses the fold game's fold" begin
    for (_, settings) in arm_variants()
        result = trace_endpoint(LogitGame(FOLD_GAME, 2.0f0), 1, settings)
        @test !is_error(result)
        if !is_error(result)
            @test residual_norm(FOLD_GAME, unwrap(result).logits, 2.0f0) < 1.0f-3
            @test only(unwrap(result).logits) > 3
        end
    end
end
