@testset "the corrector holds the relative precision at its pin, whatever the gap floor" begin
    check = GameTracing.CorrectorReach(
        GameTracing.AcceptAll(),
        GameTracing.CorrectorSettings{Float32}(),
        TraceGeometry{Float32}(),
        zeros(Float32, 3),
        zeros(Float32, 3),
        zero(Float32),
    )
    # Gaps of 1e6 put the gap floor near 4, far above a relative precision 0.04 from its pin.
    point = Float32[1.0f6, -1.0f6, 0]
    @test GameTracing.settled_tolerance(check, point) > 0.04f0
    @test !GameTracing.settled(check, (; x = point, fx = Float32[0, 0, -0.04f0], residual = 0.04f0))
    @test GameTracing.settled(check, (; x = point, fx = zeros(Float32, 3), residual = zero(Float32)))
    @test !GameTracing.settled(check, (; x = point, res_f = Float32[0, 0, -0.04f0], residual = 0.04f0))
end
