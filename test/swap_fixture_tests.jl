"""The `before` branch point at the swap and the root `level_root` reaches from it on the `after` map there."""
function swap_points(swap::MapSwap)::NamedTuple
    level = swap.precision * swap.swap_level
    before = branch_reference(swap.before, swap.width, level)
    return (; before, after = level_root(swap.after, before.logits, level))
end

@testset "map swap fixtures: the near swap moves the branch a little, the far swap leaves only a distant root" begin
    near = swap_points(NEAR_SWAP)
    @test near.before.reached
    @test 1.0e-3 < maximum(abs, near.after .- near.before.logits) < 0.2
    far = swap_points(FAR_SWAP)
    @test far.before.reached
    @test maximum(abs, far.after .- far.before.logits) > 2
end
