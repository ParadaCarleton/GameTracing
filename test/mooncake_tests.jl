using ErrorTypes: Ok, unwrap
using Mooncake: Mooncake

"""The sum of twice `point`, reached through a successful `TraceResult` that a caller must unwrap."""
function unwrapped_total(point::AbstractVector{E})::E where {E <: Real}
    result::GameTracing.TraceResult{E} = Ok(Traced(2 .* point))
    return sum(unwrap(result).logits)
end

@testset "Mooncake differentiates through an unwrapped trace result" begin
    point = Float32[0.5f0, -1.0f0]
    rule = Mooncake.build_rrule(unwrapped_total, point)
    total, (_, gradient) = Mooncake.value_and_gradient!!(rule, unwrapped_total, point)
    @test total == unwrapped_total(point)
    @test gradient == Float32[2, 2]
end
