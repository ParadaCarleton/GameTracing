using GameTracing: BranchWorkspace, branch_projection, branch_system!, projection_step
using ErrorTypes: unwrap
using LinearAlgebra: Diagonal, dot, norm, normalize, nullspace
using Random: RandomDevice, Xoshiro

"""PALC's weights `[θ / k … θ / k, 1 - θ]` over a branch point of `width` logits."""
function palc_weights(width::Integer, state_weight::E)::Diagonal{E} where {E <: Real}
    return Diagonal([fill(state_weight / width, width); 1 - state_weight])
end

"""`game`'s residual `F` and its Jacobian `J` in `[asinh(logits); ρ]` at `point`."""
function branch_equations(game::LogitGame, point::AbstractVector{E})::NamedTuple where {E <: Real}
    width = length(point) - 1
    system = branch_system!(BranchWorkspace(point), game, point, last(point))
    return (; residual = system.residual[1:width], jacobian = system.jacobian[1:width, :])
end

@testset "projection_step is the minimum-norm Newton step in PALC's norm" begin
    rng = Xoshiro(rand(RandomDevice(), UInt))
    games = (
        LogitGame(LinearGainPass(float.(GAME.interaction), float.(GAME.stand_alone)), float(TARGET_PRECISION)),
        LogitGame(ThresholdPass(float(4), float(2)), float(1)),
        LogitGame(LinearGainPass(randn(rng, 3, 3), randn(rng, 3)), float(2)),
    )
    for (game, width) in zip(games, (2, 1, 3)), state_weight in (float(1 // 5), float(1 // 2), float(9 // 10))
        for _ in 1:5
            logits = randn(rng, width) ./ 10
            relative_precision = 1 // 2 + 2 * rand(rng) / 5
            step = projection_step(game, logits, relative_precision, state_weight)
            start = branch_equations(game, [asinh.(logits); relative_precision])
            # Newton's linearized equation holds, and no part of the step runs along the branch's tangent.
            @test norm(start.jacobian * step .+ start.residual) <= inv(float(10^10)) * (1 + norm(start.residual))
            tangent = vec(nullspace(start.jacobian))
            @test abs(dot(palc_weights(width, state_weight) * step, tangent)) <= inv(float(10^10)) * (1 + norm(step))
        end
    end
end

"""The residual `branch_projection` leaves from `on_branch`'s logits moved `offset` along `direction`."""
function projected_residual(
    game::LogitGame,
    on_branch::NamedTuple,
    offset::Real,
    direction::AbstractVector,
    settings::TraceSettings,
)::typeof(float(offset))
    projected = branch_projection(game, on_branch.logits .+ offset .* direction, on_branch.relative_precision, settings)
    return norm(branch_equations(game, [asinh.(projected.logits); projected.relative_precision]).residual)
end

@testset "branch_projection converges quadratically onto the branch" begin
    rng = Xoshiro(rand(RandomDevice(), UInt))
    game = LogitGame(LinearGainPass(randn(rng, 3, 3), randn(rng, 3)), float(2))
    settings = TraceSettings{typeof(float(1))}()
    on_branch = (; logits = zeros(3), relative_precision = float(1 // 2))
    for _ in 1:10
        on_branch = branch_projection(game, on_branch.logits, on_branch.relative_precision, settings)
    end
    @test norm(branch_equations(game, [asinh.(on_branch.logits); on_branch.relative_precision]).residual) < inv(float(10^12))
    direction = normalize(randn(rng, 3))
    # Quadratic convergence drops the residual left a hundredfold per decade of offset; linear, tenfold.
    coarse = projected_residual(game, on_branch, inv(float(100)), direction, settings)
    fine = projected_residual(game, on_branch, inv(float(1000)), direction, settings)
    @test fine <= coarse / 30
end

@testset "branch_projection applies one projection_step in the trace's norm" begin
    game = LogitGame(GAME, TARGET_PRECISION)
    settings = TraceSettings{Float32}(state_weight = 3 // 10)
    logits, relative_precision = Float32[1 // 10, -1 // 5], 3.0f0 / 4
    step = projection_step(game, logits, relative_precision, settings.state_weight)
    projected = branch_projection(game, logits, relative_precision, settings)
    @test projected.logits == sinh.(asinh.(logits) .+ step[1:WIDTH])
    @test projected.relative_precision == relative_precision + last(step)
    @test projected.logits isa Vector{Float32}
end

@testset "projection_step rejects a state weight outside (0, 1)" begin
    for state_weight in (0.0f0, 1.0f0, -0.5f0, 1.5f0, NaN32)
        @test_throws ArgumentError projection_step(LogitGame(GAME, TARGET_PRECISION), zeros(Float32, WIDTH), 0.5f0, state_weight)
    end
end

@testset "branch_projection leaves a point on the branch where it is" begin
    traced = unwrap(trace_endpoint(LogitGame(GAME, TARGET_PRECISION), WIDTH, TraceSettings{Float32}()))
    projected = branch_projection(LogitGame(GAME, TARGET_PRECISION), traced.logits, 1.0f0, TraceSettings{Float32}())
    @test maximum(abs, projected.logits .- traced.logits) < 1.0f-4
    @test abs(projected.relative_precision - 1) < 1.0f-4
end

@testset "near a fold, where a correction at fixed precision is singular, the projection converges quadratically" begin
    # On ThresholdPass(a, t) the branch is precision = z / (a σ(z - t)); it folds where z (1 - σ(z - t)) = 1.
    amplitude, threshold, target = float(6), float(3), float(2)
    sigmoid(logit::Real) = inv(1 + exp(-logit))
    low, high = float(1), float(2)
    for _ in 1:60
        middle = (low + high) / 2
        if middle * (1 - sigmoid(middle - threshold)) < 1
            low = middle
        else
            high = middle
        end
    end
    fold = (low + high) / 2
    fold_precision = fold / (amplitude * sigmoid(fold - threshold)) / target
    game = LogitGame(ThresholdPass(amplitude, threshold), target)
    # At the fold the derivative in the logit vanishes, so Newton at fixed precision has no step.
    at_fold = branch_equations(game, [asinh(fold); fold_precision])
    @test abs(only(at_fold.residual)) <= inv(float(10^10))
    @test abs(first(at_fold.jacobian)) <= inv(float(10^8)) * norm(at_fold.jacobian)
    # Newton at fixed precision meets a double root here and converges only linearly; the projection stays quadratic.
    settings = TraceSettings{typeof(float(1))}(state_weight = 2 // 5)
    on_branch = (; logits = [fold], relative_precision = fold_precision)
    for direction in ([-one(fold)], [one(fold)])
        coarse = projected_residual(game, on_branch, inv(float(100)), direction, settings)
        fine = projected_residual(game, on_branch, inv(float(1000)), direction, settings)
        @test fine <= coarse / 30
    end
end
