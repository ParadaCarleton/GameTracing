using ErrorTypes: Ok, is_error, unwrap, unwrap_error
using Mooncake: Mooncake

"""The sum of twice `point`, reached through a successful `TraceResult` that a caller must unwrap."""
function unwrapped_total(point::AbstractVector{E})::E where {E <: Real}
    result::GameTracing.TraceResult{E} = Ok(Traced(2 .* point, 0))
    return sum(unwrap(result).logits)
end

@testset "a trace result is unusable without handling its failure" begin
    failed = trace_endpoint(LogitGame(PoisonedPass(GAME, 0.0f0), TARGET_PRECISION), WIDTH, TraceSettings{Float32}())
    @test failed isa GameTracing.TraceResult{Float32}
    @test is_error(failed)
    @test_throws ErrorException unwrap(failed)
    @test unwrap_error(failed) isa TraceFailure
    traced = trace_endpoint(LogitGame(GAME, TARGET_PRECISION), WIDTH, TraceSettings{Float32}())
    @test_throws ErrorException unwrap_error(traced)
end

@testset "Mooncake differentiates through an unwrapped trace result" begin
    point = Float32[0.5f0, -1.0f0]
    rule = Mooncake.build_rrule(unwrapped_total, point)
    total, (_, gradient) = Mooncake.value_and_gradient!!(rule, unwrapped_total, point)
    @test total == unwrapped_total(point)
    @test gradient == Float32[2, 2]
end
