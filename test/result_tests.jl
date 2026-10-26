using ErrorTypes: is_error, unwrap, unwrap_error

@testset "a trace result is unusable without handling its failure" begin
    failed = trace_endpoint(LogitGame(PoisonedPass(GAME, 0.0f0), TARGET_PRECISION), WIDTH, TraceSettings{Float32}())
    @test failed isa GameTracing.TraceResult{Float32}
    @test is_error(failed)
    @test_throws ErrorException unwrap(failed)
    @test unwrap_error(failed) isa TraceFailure
    traced = trace_endpoint(LogitGame(GAME, TARGET_PRECISION), WIDTH, TraceSettings{Float32}())
    @test_throws ErrorException unwrap_error(traced)
end

