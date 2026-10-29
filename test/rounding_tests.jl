using ErrorTypes: is_error, unwrap

const DOUBLE = typeof(float(1))

@testset "rounding_bound is Newton's limiting accuracy 2ρ‖(∂F/∂w)⁻¹‖∞(3u‖w‖∞ + gain_error)" begin
    # ∂F/∂w = [1 1; 0 1] has inverse [1 -1; 0 1], of infinity norm 2; the precision column and row play no part.
    jacobian = Float32[1 1 9; 0 1 9; 0 0 1]
    point, gain_error = Float32[3, -6, 0.25f0], 1.0f-5
    expected = 2 * 0.25f0 * 2 * (3 * eps(Float32) / 2 * 6 + gain_error)
    @test GameTracing.rounding_bound(jacobian, point, gain_error) ≈ expected rtol = 1.0f-5
    @test GameTracing.rounding_bound(jacobian, point, 2 * gain_error) > GameTracing.rounding_bound(jacobian, point, gain_error)
    # A pass without `gain_rounding` allows no rounding, whatever the bound.
    check = GameTracing.CorrectorReach(
        GameTracing.AcceptAll(), CorrectorSettings{Float32}(), TraceGeometry{Float32}(), point, point, 1.0f0,
    )
    @test !GameTracing.within_rounding(check, (; x = point, J = jacobian), 0.0f0)
    allowing = GameTracing.CorrectorReach(
        GameTracing.AcceptAll(), CorrectorSettings{Float32}(), TraceGeometry{Float32}(), point, point, 1.0f0, gain_error,
    )
    @test GameTracing.within_rounding(allowing, (; x = point, J = jacobian), expected)
    @test !GameTracing.within_rounding(allowing, (; x = point, J = jacobian), 2 * expected)
    @test !GameTracing.within_rounding(allowing, (; x = point, J = nothing), 0.0f0)
end

@testset "gain_rounding is zero by default and 2 γ_(K+1) max|payoff| for a normal-form game" begin
    @test GameTracing.gain_rounding(GAME, Float32) == 0
    payoffs = (Float32[1 -4; 2 0; 3 1], Float32[0 2; -1 1; 5 0])
    # Strategy counts (3, 2): K + 1 = 6 + 2 (3 + 3) + 1 = 19 operations.
    unit = eps(Float32) / 2
    @test gain_rounding(NormalFormGains(payoffs), Float32) ≈ 2 * (19 * unit / (1 - 19 * unit)) * 5 rtol = 1.0f-5
end

"""The largest gain error of `pass` evaluated in Float32 at `logits` against the same pass in double precision."""
function gain_error_measured(single::Any, double::Any, logits::AbstractVector{Float32})::DOUBLE
    width = length(logits)
    gains = zeros(Float32, width)
    single(gains, zeros(Float32, width, width), zeros(Float32, width), logits, 1.0f0)
    reference = zeros(DOUBLE, width)
    double(reference, zeros(DOUBLE, width, width), zeros(DOUBLE, width), logits .* float(1), float(1))
    return maximum(abs, gains .- reference)
end

@testset "gain_rounding bounds the rounding error each pass's gains make in Float32" begin
    rng = fresh_rng()
    for counts in ((2, 2), (3, 3), (2, 2, 2), (2, 3, 4)), _ in 1:20
        payoffs = ntuple(_ -> 3 .* randn(rng, Float32, counts), length(counts))
        single, double = NormalFormGains(payoffs), NormalFormGains(map(table -> table .* float(1), payoffs))
        logits = 4 .* randn(rng, Float32, logit_count(single))
        @test gain_error_measured(single, double, logits) <= gain_rounding(single, Float32)
    end
    for players in 2:4, _ in 1:20
        payoffs = ntuple(_ -> 3 .* randn(rng, Float32, ntuple(_ -> 2, players)), players)
        single, double = NormalFormPass(payoffs), NormalFormPass(map(table -> table .* float(1), payoffs))
        logits = 4 .* randn(rng, Float32, players)
        @test gain_error_measured(single, double, logits) <= gain_rounding(single, Float32)
    end
end

@testset "at a pure equilibrium far past ulp(z) = logit_tolerance, the trace accepts rounding noise and reports it" begin
    gains = NormalFormGains(map(table -> Float32.(table), LOGITNASH_GAMES.prisoners_dilemma))
    target = 1.0f5
    settings = TraceSettings{Float32}(; geometry = TraceGeometry{Float32}(; distance = FisherLogPrecisionDistance))
    game = LogitGame(gains, target)
    result = trace_endpoint(game, logit_count(gains), settings)
    @test !is_error(result)
    if !is_error(result)
        traced = unwrap(result)
        # Both players defect, their second strategy, at logits near 1e5 whose ulp passes the tolerance.
        @test all(strategy -> last(strategy) > 1 - 1.0f-3, mixed_strategies(gains, traced.logits))
        @test traced.logit_accuracy > settings.corrector.logit_tolerance
        point = [traced.logits; 1.0f0]
        jacobian = GameTracing.branch_system!(GameTracing.BranchWorkspace(game, point), game, point, 1.0f0).jacobian
        @test traced.logit_accuracy <= GameTracing.rounding_bound(jacobian, point, target * gain_rounding(gains, Float32))
        @info "pure equilibrium past ulp" traced.logit_accuracy traced.steps
    end
end
