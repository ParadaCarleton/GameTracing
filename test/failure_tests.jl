const FAILURE_SETTINGS = TraceSettings{Float32}(first_step = 0.01f0, max_step = 0.05f0)

@testset "a non-finite response past a precision stops the trace short of it" begin
    result = trace_endpoint(
        LogitGame(PoisonedPass(GAME, 1.0f0), TARGET_PRECISION),
        WIDTH,
        FAILURE_SETTINGS,
    )
    @test result isa TraceFailure
    @test result.cause == GameTracing.TraceStuck
    @test TARGET_PRECISION * result.context.relative_precision <= 1.0f0
end

@testset "a non-finite Jacobian is a failure value" begin
    result = trace_endpoint(
        LogitGame(NaNSlopePass(GAME), TARGET_PRECISION),
        WIDTH,
        FAILURE_SETTINGS,
    )
    @test result isa TraceFailure
    @test result.cause in (GameTracing.Singular, GameTracing.TraceStuck)
end

@testset "a caller's exception propagates" begin
    @test_throws ArgumentError trace_endpoint(
        LogitGame(ThrowingPass(GAME), TARGET_PRECISION),
        WIDTH,
        FAILURE_SETTINGS,
    )
end

struct RefuseConverged end

(::RefuseConverged)(::AbstractVector, ::Real, ::Bool)::Bool = false

@testset "refusing every corrector ends the trace" begin
    result = trace_endpoint(
        LogitGame(GAME, TARGET_PRECISION),
        WIDTH,
        FAILURE_SETTINGS;
        accept_corrected = RefuseConverged(),
    )
    @test result isa TraceFailure
    @test result.cause == GameTracing.TraceStuck
end

@testset "a corrector reach too short to hold any step ends the trace" begin
    result = trace_endpoint(LogitGame(FOLD_GAME, 2.0f0), 1, TraceSettings{Float32}(
        state_weight = 0.4f0,
        corrector_reach = 1.0f-6,
        first_step = 0.05f0,
        max_step = 0.1f0,
    );
    )
    @test result isa TraceFailure
    @test result.cause in (GameTracing.TraceStuck, GameTracing.NoConvergence)
end

@testset "exceptions BifurcationKit and LAPACK throw are told apart from a caller's" begin
    @test GameTracing.uncorrected_start("Stopping continuation.")
    @test GameTracing.uncorrected_start(ErrorException(
        "Newton failed to converge at the start",
    ))
    @test !GameTracing.uncorrected_start(ErrorException("another"))
    @test !GameTracing.uncorrected_start("another")
    @test !GameTracing.uncorrected_start(ArgumentError("bad game"))
    @test GameTracing.solver_blowup(SingularException(2))
    @test GameTracing.solver_blowup(LAPACKException(2))
    @test GameTracing.solver_blowup(ArgumentError("matrix contains Infs or NaNs"))
    @test GameTracing.solver_blowup(ArgumentError("LAPACK failed"))
    @test !GameTracing.solver_blowup(ArgumentError("bad game"))
    @test !GameTracing.solver_blowup(DimensionMismatch("bad game"))
end

@testset "a bordered matrix of the wrong size is rejected" begin
    @test_throws DimensionMismatch GameTracing.bordered_matrix!(
        zeros(Float32, 2, 2),
        ones(Float32, 2, 2),
        ones(Float32, 2),
        ones(Float32, 2),
        1.0f0,
        1.0f0,
        1.0f0,
        nothing,
    )
end

@testset "a triangular bordered system solves like the dense one" begin
    solver = GameTracing.matrix_bls(2, Float32)
    jacobian = Float32[2 0; 0 3]
    border = Float32[0, 0]
    solution, parameter, _, _ = solver(
        jacobian,
        border,
        Float32[0, 0],
        1.0f0,
        Float32[2, 3],
        1.0f0,
    )
    @test collect(solution) ≈ Float32[1, 1]
    @test parameter ≈ 1.0f0
end

@testset "a predictor that did not move has no reach to exceed" begin
    check = GameTracing.CorrectorReach(
        GameTracing.AcceptAll(),
        2.0f0,
        0.5f0,
        1.0f-5,
        32.0f0,
        zeros(Float32, 3),
        0.0f0,
    )
    @test GameTracing.within_reach(check, ones(Float32, 3))
end

@testset "a game that is non-finite at every positive precision fails at the first step" begin
    result = trace_endpoint(
        LogitGame(PoisonedPass(GAME, 0.0f0), TARGET_PRECISION),
        WIDTH,
        FAILURE_SETTINGS,
    )
    @test result isa TraceFailure
    @test result.cause == GameTracing.TraceStuck
    @test result.context.steps == 0
end

@testset "BifurcationKit's report of an uncorrectable start is a stuck trace" begin
    progress = GameTracing.TraceProgress(
        zeros(Float32, 2),
        0.0f0,
        0,
        0.0f0,
        0.0f0,
        zeros(Float32, 2),
        0.0f0,
        false,
    )
    result = GameTracing.reached_state(
        progress,
        (; step_length = 0.0f0, exception = "Stopping continuation."),
        1.0f0,
        10,
        TARGET_PRECISION,
        (;),
    )
    @test result isa TraceFailure
    @test result.cause == GameTracing.TraceStuck
end
