using LinearAlgebra: SingularException, tril, triu
using Random: RandomDevice, Xoshiro

"""A random bordered system of `width` unknowns: its Jacobian, border column, border row, corner, and the two right-hand sides."""
function bordered_system(rng::Xoshiro, width::Integer, ::Type{E})::NamedTuple where {E <:
    Real}
    return (;
        jacobian = randn(rng, E, width, width) +
            E[row == column ? 4 : 0 for row in 1:width, column in 1:width],
        border_column = randn(rng, E, width),
        border_row = randn(rng, E, width),
        corner = randn(rng, E),
        residual = randn(rng, E, width),
        border_value = randn(rng, E),
    )
end

"""The bordered solver `BifurcationKit.MatrixBLS` and ours return on `system`, as `(; reference, ours)` of `(x, scalar)`."""
function bordered_solutions(
    system::NamedTuple,
    row_scale::Real,
    corner_scale::Real,
)::NamedTuple
    width = length(system.residual)
    ours = GameTracing.matrix_bls(width, eltype(system.jacobian))(
        system.jacobian,
        system.border_column,
        system.border_row,
        system.corner,
        system.residual,
        system.border_value,
        row_scale,
        corner_scale,
    )
    reference = GameTracing.BifurcationKit.MatrixBLS()(
        system.jacobian,
        system.border_column,
        system.border_row,
        system.corner,
        system.residual,
        system.border_value,
        row_scale,
        corner_scale,
    )
    return (; reference, ours)
end

"""Whether our bordered solve on `system` is bit-identical to `MatrixBLS`'s, as `InPlaceMatrixBLS` promises."""
function bit_identical(system::NamedTuple, row_scale::Real, corner_scale::Real)::Bool
    solutions = bordered_solutions(system, row_scale, corner_scale)
    return collect(solutions.ours[1]) == collect(solutions.reference[1]) &&
        solutions.ours[2:4] == solutions.reference[2:4]
end

@testset "the in-place bordered solver is bit-identical to BifurcationKit's MatrixBLS" begin
    rng = Xoshiro(rand(RandomDevice(), UInt))
    for width in 1:5, (row_scale, corner_scale) in ((1.0f0, 1.0f0), (0.3f0, 2.0f0))
        @test bit_identical(bordered_system(rng, width, Float32), row_scale, corner_scale)
    end
end

# LU and triangular substitution round differently, so only the triangular path matches `MatrixBLS`'s `\`.
@testset "a triangular bordered matrix is solved bit-identically to MatrixBLS's triangular solve" begin
    rng = Xoshiro(rand(RandomDevice(), UInt))
    for width in 1:5, _ in 1:20
        system = bordered_system(rng, width, Float32)
        lower = (; system..., jacobian = tril(system.jacobian), border_column = zeros(Float32, width))
        upper = (; system..., jacobian = triu(system.jacobian), border_row = zeros(Float32, width))
        @test bit_identical(lower, 0.3f0, 2.0f0)
        @test bit_identical(upper, 0.3f0, 2.0f0)
    end
end

"""The exception `solver` throws on `system`'s bordered solve, or `nothing` when it returns."""
function solve_exception(solver::Any, system::NamedTuple)::Union{Nothing, Exception}
    try
        solver(
            system.jacobian,
            system.border_column,
            system.border_row,
            system.corner,
            system.residual,
            system.border_value,
        )
    catch exception
        return exception
    end
    return nothing
end

@testset "a singular or non-finite bordered matrix throws what MatrixBLS throws" begin
    singular = (;
        jacobian = Float32[1 2; 2 4],
        border_column = Float32[1, 2],
        border_row = Float32[1, 1],
        corner = 1.0f0,
        residual = Float32[1, 1],
        border_value = 1.0f0,
    )
    non_finite = (; singular..., jacobian = Float32[NaN32 2; 2 4])
    # A zero first column makes the first pivot the zero one (LAPACK `info == 1`).
    zero_first_column = (; singular..., jacobian = Float32[0 2; 0 4], border_row = Float32[0, 1])
    cases = ((singular, SingularException), (zero_first_column, SingularException), (non_finite, ArgumentError))
    for (system, expected) in cases
        @test solve_exception(GameTracing.matrix_bls(2, Float32), system) isa expected
        @test solve_exception(GameTracing.BifurcationKit.MatrixBLS(), system) isa expected
    end
end

@testset "the in-place determinant sign agrees with MatrixBLS's, on singular and non-finite matrices too" begin
    kit = GameTracing.BifurcationKit
    settings = TraceSettings{Float32}()
    start = zeros(Float32, WIDTH + 1)
    problem = GameTracing.branch_problem(
        LogitGame(GAME, TARGET_PRECISION),
        settings.coordinates,
        start,
        0.0f0;
        record_from_solution = GameTracing.trace_record_precision,
    )
    iter = kit.ContIterable(
        problem,
        GameTracing.palc_algorithm(settings, WIDTH),
        GameTracing.continuation_options(settings);
        verbosity = 0,
    )
    state, _ = iterate(iter)
    ours = GameTracing.matrix_bls(WIDTH + 1, Float32)
    rng = Xoshiro(rand(RandomDevice(), UInt))
    for _ in 1:50
        jacobian = randn(rng, Float32, WIDTH + 1, WIDTH + 1)
        slope = randn(rng, Float32, WIDTH + 1)
        @test kit.bordered_determinant_sign(ours, iter, state, jacobian, slope) ==
            kit.bordered_determinant_sign(kit.MatrixBLS(), iter, state, jacobian, slope)
    end
    dependent_slope = Float32[1, 2, 0]
    for degenerate in (Float32[1 2 0; 2 4 0; 0 0 1], Float32[NaN32 2 0; 2 4 0; 0 0 1])
        @test kit.bordered_determinant_sign(ours, iter, state, degenerate, dependent_slope) ==
            kit.bordered_determinant_sign(kit.MatrixBLS(), iter, state, degenerate, dependent_slope)
    end
end

@testset "bordered_matrix! lays out the Jacobian, border column, scaled border row and corner" begin
    bordered = zeros(Float32, 3, 3)
    GameTracing.bordered_matrix!(
        bordered,
        Float32[1 2; 3 4],
        Float32[5, 6],
        Float32[7, 8],
        9.0f0,
        2.0f0,
        3.0f0,
        nothing,
    )
    @test bordered == Float32[1 2 5; 3 4 6; 14 16 27]
    GameTracing.bordered_matrix!(
        bordered,
        Float32[1 2; 3 4],
        Float32[5, 6],
        Float32[7, 8],
        9.0f0,
        2.0f0,
        3.0f0,
        row -> row .*= 10,
    )
    @test bordered[3, :] == Float32[140, 160, 27]
    @test_throws DimensionMismatch GameTracing.bordered_matrix!(
        zeros(Float32, 4, 4),
        Float32[1 2; 3 4],
        Float32[5, 6],
        Float32[7, 8],
        9.0f0,
        2.0f0,
        3.0f0,
        nothing,
    )
end

@testset "the bordering solver choice builds BifurcationKit's BorderingBLS" begin
    @test GameTracing.bordered_linear_solver(BorderingSolve, WIDTH + 1, Float32) isa
        GameTracing.BifurcationKit.BorderingBLS
    @test GameTracing.bordered_linear_solver(MatrixSolve, WIDTH + 1, Float32) isa
        GameTracing.InPlaceMatrixBLS
end
