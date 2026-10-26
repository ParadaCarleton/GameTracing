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

@testset "augmented_theta rejects a trace PALC cannot weight" begin
    @test GameTracing.augmented_theta(0.5f0, 3) ≈ 0.5f0 * 4 / 3
    @test_throws ArgumentError GameTracing.augmented_theta(0.9f0, 3)
end

@testset "an element type LAPACK cannot factor gets BifurcationKit's matrix solver" begin
    @test GameTracing.matrix_bls(WIDTH, Float16) isa GameTracing.BifurcationKit.MatrixBLS
    @test GameTracing.matrix_bls(WIDTH, Float32) isa GameTracing.InPlaceMatrixBLS
end

@testset "PALC orients by determinant only where the solver has one" begin
    for check in (true, false), solver in instances(BorderedSolver)
        settings = TraceSettings{Float32}(
            orientation_check = check,
            bordered_solver = solver,
        )
        oriented = GameTracing.palc_algorithm(settings, WIDTH).orientation_check
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
        options = GameTracing.continuation_options(settings)
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

@testset "the fused system reuses one workspace and matches a fresh evaluation in every coordinate system" begin
    game = LogitGame(GAME, TARGET_PRECISION)
    point = Float32[0.3f0, -0.2f0, 0.6f0]
    for coordinates in instances(TraceCoordinates)
        workspace = GameTracing.BranchWorkspace(point)
        system = GameTracing.GameSystem(game, coordinates, workspace)
        residual_at = GameTracing.GameResidual(game, coordinates)
        first_residual, first_jacobian = system(point, 0.5f0)
        second_residual, second_jacobian = system(point .+ 0.1f0, 0.5f0)
        @test first_residual === second_residual === workspace.residual
        @test first_jacobian === second_jacobian === workspace.jacobian
        residual, jacobian = system(point, 0.5f0)
        @test residual == residual_at(point, 0.5f0)
        @test jacobian == GameTracing.GameJacobian(game, coordinates, GameTracing.BranchWorkspace(point))(point, 0.5f0)
        # A central difference in each coordinate, the relative precision included, checks the Jacobian formed in place.
        step = 1.0f-3
        for column in eachindex(point)
            nudge = step .* (eachindex(point) .== column)
            slope = (residual_at(point .+ nudge, 0.5f0) .- residual_at(point .- nudge, 0.5f0)) ./ (2 * step)
            @test maximum(abs, slope .- jacobian[:, column]) < 1.0f-2
        end
    end
end

@testset "every coordinate system writes the same logit residual, up to its own scale" begin
    game = LogitGame(GAME, TARGET_PRECISION)
    logits, relative_precision = Float32[0.4f0, -0.3f0], 0.6f0
    for coordinates in instances(TraceCoordinates)
        point = [GameTracing.trace_state(coordinates, logits, relative_precision); relative_precision]
        @test GameTracing.trace_logits(coordinates, point[1:WIDTH], relative_precision) ≈ logits
        residual = GameTracing.GameResidual(game, coordinates)(point, relative_precision)[1:WIDTH]
        # Payoff gaps divide the logit residual `z - λ G` by `ρ`.
        scale = ifelse(coordinates == PayoffGaps, relative_precision, 1.0f0)
        @test scale .* residual ≈ logits .- evaluated(GAME, logits, TARGET_PRECISION * relative_precision).response
    end
    @test_throws ArgumentError GameTracing.trace_state(PayoffGaps, logits, 0.0f0)
end

@testset "payoff gaps start at the target's flat-start gains" begin
    start = GameTracing.trace_start(LogitGame(GAME, TARGET_PRECISION), PayoffGaps, WIDTH, Float32)
    @test start[1:WIDTH] ≈ TARGET_PRECISION .* evaluated(GAME, zeros(Float32, WIDTH), 1.0f0).response
    @test last(start) == 0
    @test GameTracing.trace_start(LogitGame(GAME, TARGET_PRECISION), Logits, WIDTH, Float32) == zeros(Float32, WIDTH + 1)
end

@testset "a Jacobian call leaves the residual the fused system returned" begin
    bifurcation_kit = GameTracing.BifurcationKit
    point = Float32[0.3f0, -0.2f0, 0.6f0]
    problem = GameTracing.branch_problem(
        LogitGame(GAME, TARGET_PRECISION),
        AsinhLogits,
        point,
        0.5f0;
        record_from_solution = GameTracing.trace_record_precision,
    )
    residual, _ = bifurcation_kit.residual_and_jacobian(problem, point, 0.5f0)
    kept = copy(residual)
    bifurcation_kit.jacobian(problem, point .+ 0.1f0, 0.5f0)
    @test residual == kept
end
