using BifurcationKit: BifurcationKit
using ErrorTypes: is_error, unwrap, unwrap_error
using Random: Random

@testset "trace reaches a fixed point of the logit map" begin
    result = trace_endpoint(
        LogitGame(GAME, TARGET_PRECISION),
        WIDTH,
        TraceSettings{Float32}(),
    )
    @test !is_error(result)
    @test residual_norm(GAME, unwrap(result).logits, TARGET_PRECISION) < 1.0f-4
end

@testset "tangent and bordered solver choices reach the same unique equilibrium" begin
    game = LogitGame(GAME, TARGET_PRECISION)
    reference = unwrap(trace_endpoint(game, WIDTH, TraceSettings{Float32}())).logits
    for settings in (
            TraceSettings{Float32}(tangent = BifurcationKit.Secant()),
            TraceSettings{Float32}(bordered_solver = BorderingSolve),
            TraceSettings{Float32}(step_control = nothing),
            TraceSettings{Float32}(step_control = StepControl{Float32}(
                distance_limit = nothing,
                contraction_limit = nothing,
            )),
        )
        traced = trace_endpoint(game, WIDTH, settings)
        @test !is_error(traced)
        @test maximum(abs, unwrap(traced).logits .- reference) < 1.0f-3
    end
end

@testset "a spent step budget is a failure value" begin
    result = trace_endpoint(
        LogitGame(GAME, TARGET_PRECISION),
        WIDTH,
        TraceSettings{Float32}(max_steps = 1, first_step = 0.005f0, max_step = 0.01f0),
    )
    @test is_error(result)
    @test unwrap_error(result).cause in (GameTracing.NoConvergence, GameTracing.TraceStuck)
end

@testset "a refusing keep_going callback stops the trace" begin
    refuse_all(::AbstractVector, ::Real) = false
    result = trace_endpoint(
        LogitGame(GAME, TARGET_PRECISION),
        WIDTH,
        TraceSettings{Float32}();
        keep_going = refuse_all,
    )
    @test unwrap_error(result).cause == GameTracing.TraceStuck
    @test unwrap_error(result).context.relative_precision < 1
    @test residual_norm(
        GAME,
        unwrap_error(result).context.logits,
        TARGET_PRECISION * unwrap_error(result).context.relative_precision,
    ) <
        1.0f-4
end

"""A `keep_going` callback that goes on only while the point it sees solves the game's map at its precision."""
struct OnTheBranch{P, E <: Real}
    pass::P
    target_precision::E
end

function (probe::OnTheBranch)(logits::AbstractVector, relative_precision::Real)::Bool
    return residual_norm(probe.pass, logits, probe.target_precision * relative_precision) <
        1.0f-4
end

"""A `keep_going` callback that stops the trace once its relative precision passes `limit`."""
struct StopPast{E <: Real}
    limit::E
end

(stop::StopPast)(::AbstractVector, relative_precision::Real)::Bool = relative_precision < stop.limit

@testset "a failure reports its tangent in logits" begin
    result = trace_endpoint(
        LogitGame(GAME, TARGET_PRECISION),
        WIDTH,
        TraceSettings{Float32}();
        keep_going = StopPast(0.75f0),
    )
    @test unwrap_error(result).cause == GameTracing.TraceStuck
    (; logits, logit_tangent, tangent_precision, relative_precision) = unwrap_error(result).context
    # A step `h` along the branch's tangent leaves a residual of order h², far under h ‖dz‖.
    step = 1.0f-2
    moved = residual_norm(GAME, logits .+ step .* logit_tangent, TARGET_PRECISION * (relative_precision + step * tangent_precision))
    @test moved < step * maximum(abs, logit_tangent) / 10
end

@testset "callbacks see logits" begin
    result = trace_endpoint(
        LogitGame(GAME, TARGET_PRECISION),
        WIDTH,
        TraceSettings{Float32}();
        keep_going = OnTheBranch(GAME, TARGET_PRECISION),
    )
    @test !is_error(result)
end

@testset "a non-finite game fails at the start" begin
    broken = LinearGainPass(Float32[NaN32 1; 1 0], Float32[0.5f0, -0.25f0])
    result = trace_endpoint(
        LogitGame(broken, TARGET_PRECISION),
        WIDTH,
        TraceSettings{Float32}();
        context = (; label = :broken),
    )
    @test unwrap_error(result).cause == GameTracing.NotFinite
    @test unwrap_error(result).context.label == :broken
end

@testset "every state weight in (0, 1) traces, a one-logit game at the default settings included" begin
    for state_weight in (0.1f0, 0.5f0, 0.9f0), distance in instances(StepDistance)
        geometry = TraceGeometry{Float32}(; distance, state_weight)
        # A log-precision trace carries its state weight in its dot and runs PALC at a weight set by the target.
        palc_state_weight = ifelse(distance == FisherLogPrecisionDistance, GameTracing.log_precision_palc_state_weight(2.0f0), state_weight)
        @test GameTracing.palc_geometry(geometry, 1, 2.0f0).palc_state_weight == palc_state_weight
    end
    for state_weight in (0.0f0, 1.0f0, -0.5f0, NaN32)
        @test_throws ArgumentError GameTracing.palc_geometry(TraceGeometry{Float32}(; state_weight), 1, 2.0f0)
    end
    result = trace_endpoint(LogitGame(FOLD_GAME, 2.0f0), 1, TraceSettings{Float32}())
    @test !is_error(result)
    @test residual_norm(FOLD_GAME, unwrap(result).logits, 2.0f0) < 1.0f-3
end

@testset "an element type LAPACK cannot factor gets BifurcationKit's matrix solver" begin
    @test GameTracing.matrix_bls(WIDTH, Float16) isa GameTracing.BifurcationKit.MatrixBLS
    @test GameTracing.matrix_bls(WIDTH, Float32) isa GameTracing.InPlaceMatrixBLS
end

@testset "the default smallest step is the smallest normal number of the settings' element type" begin
    for element in (Float16, Float32, Float64)
        @test TraceSettings{element}().min_step == floatmin(element)
    end
end

@testset "PALC orients by determinant only where the solver has one" begin
    for check in (true, false), solver in instances(BorderedSolver)
        settings = TraceSettings{Float32}(
            orientation_check = check,
            bordered_solver = solver,
        )
        oriented = GameTracing.palc_algorithm(settings, WIDTH, TARGET_PRECISION).orientation_check
        @test oriented == (check && solver == MatrixSolve)
    end
end

@testset "continuation options carry the settings and the trace's endpoint-only needs" begin
    rng = Random.Xoshiro(7)
    for _ in 1:20
        settings = TraceSettings{Float32}(
            first_step = rand(rng, Float32),
            min_step = 1.0f-6 * rand(rng, Float32),
            max_step = 1 + rand(rng, Float32),
            aggressiveness = rand(rng, Float32),
            first_tangent_divisor = 1 + 100 * rand(rng, Float32),
            max_steps = rand(rng, 1:50),
            corrector = CorrectorSettings{Float32}(
                residual_tolerance = 1.0f-3 * rand(rng, Float32),
                newton_iterations = rand(rng, 1:20),
            ),
        )
        options = GameTracing.continuation_options(settings, settings.first_step, settings.max_steps, settings.first_tangent_divisor)
        @test (options.ds, options.dsmin, options.dsmax, options.a, options.η) ==
            (
                settings.first_step,
                settings.min_step,
                settings.max_step,
                settings.aggressiveness,
                settings.first_tangent_divisor,
            )
        @test (options.p_min, options.p_max) == (0, 1)
        @test options.max_steps == settings.max_steps
        @test options.newton_options.tol == settings.corrector.residual_tolerance
        @test options.newton_options.max_iterations == settings.corrector.newton_iterations
        @test !options.detect_fold
        @test options.detect_bifurcation == 0
        @test options.save_sol_every_step == 0
    end
end

@testset "the fused system reuses one workspace and matches a fresh evaluation" begin
    game = LogitGame(GAME, TARGET_PRECISION)
    point = Float32[0.3f0, -0.2f0, 0.6f0]
    workspace = GameTracing.BranchWorkspace(game, point)
    system = GameTracing.GameSystem(game, workspace)
    first_residual, first_jacobian = system(point, 0.5f0)
    second_residual, second_jacobian = system(point .+ 0.1f0, 0.5f0)
    @test first_residual === second_residual === workspace.residual
    @test first_jacobian === second_jacobian === workspace.jacobian
    residual, jacobian = system(point, 0.5f0)
    @test residual == GameTracing.GameResidual(game)(point, 0.5f0)
    @test jacobian == GameTracing.GameJacobian(game, GameTracing.BranchWorkspace(game, point))(point, 0.5f0)
    # A forward difference in each coordinate checks the Jacobian the pass formed in place.
    for (column, step) in pairs(Float32[1.0f-3, 1.0f-3, 1.0f-3])
        moved = copy(point)
        moved[column] += step
        slope = (GameTracing.GameResidual(game)(moved, 0.5f0) .- GameTracing.GameResidual(game)(point, 0.5f0)) ./ step
        @test maximum(abs, slope .- jacobian[:, column]) < 1.0f-2
    end
end

@testset "the trace starts at the gaps the flat logits face, where the system is regular" begin
    pass = ThresholdPass(6.0f0, 3.0f0)
    game = LogitGame(pass, 2.0f0)
    start = GameTracing.trace_start(game, zeros(Float32, 1))
    gains = evaluated(pass, zeros(Float32, 1), 1.0f0).precision_slope
    @test GameTracing.payoff_gaps(start) ≈ 2.0f0 .* gains
    @test last(start) == 0
    residual, jacobian = GameTracing.GameSystem(game, GameTracing.BranchWorkspace(game, start))(start, 0.0f0)
    @test maximum(abs, residual) <= 4 * eps(Float32) * maximum(abs, start)
    # At zero precision the gaps decouple from the logits: ∂F/∂w is the identity.
    @test jacobian[1, 1] == 1
    @test GameTracing.logits_at(start) == zeros(Float32, 1)
end

@testset "the gap system's Jacobian matches central differences where the gains move with the precision" begin
    pass = DampedGainPass(LinearGainPass(Float32[1 -2; 0.5f0 1], Float32[0.5f0, -0.25f0]), 0.8f0)
    game = LogitGame(pass, 3.0f0)
    for point in (Float32[0.4f0, -0.7f0, 0], Float32[0.3f0, -0.2f0, 0.6f0], Float32[-1.5f0, 2, 1])
        _, jacobian = GameTracing.GameSystem(game, GameTracing.BranchWorkspace(game, point))(point, last(point))
        for column in eachindex(point)
            step = 1.0f-2
            nudge = setindex!(zeros(Float32, length(point)), step, column)
            upper = GameTracing.GameResidual(game)(point .+ nudge, last(point))
            lower = GameTracing.GameResidual(game)(point .- nudge, last(point))
            # Truncation h² M₃ / 6 and rounding eps / h each stay under 1e-3 of the Jacobian's scale here.
            @test maximum(abs, (upper .- lower) ./ 2step .- jacobian[:, column]) < 1.0f-3 * (1 + maximum(abs, jacobian))
        end
    end
    traced = unwrap(trace_endpoint(game, 2, TraceSettings{Float32}()))
    @test residual_norm(pass, traced.logits, 3.0f0) < 1.0f-4
end

@testset "a Jacobian call leaves the residual the fused system returned" begin
    bifurcation_kit = GameTracing.BifurcationKit
    point = Float32[0.3f0, -0.2f0, 0.6f0]
    problem = GameTracing.branch_problem(
        LogitGame(GAME, TARGET_PRECISION),
        point,
        0.5f0;
        record_from_solution = GameTracing.trace_record_precision,
    )
    residual, _ = bifurcation_kit.residual_and_jacobian(problem, point, 0.5f0)
    kept = copy(residual)
    bifurcation_kit.jacobian(problem, point .+ 0.1f0, 0.5f0)
    @test residual == kept
end
