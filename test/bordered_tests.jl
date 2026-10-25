using LinearAlgebra: SingularException
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

@testset "the in-place bordered solver agrees with BifurcationKit's MatrixBLS" begin
    rng = Xoshiro(rand(RandomDevice(), UInt))
    for width in 1:5, (row_scale, corner_scale) in ((1.0f0, 1.0f0), (0.3f0, 2.0f0))
        solutions = bordered_solutions(
            bordered_system(rng, width, Float32),
            row_scale,
            corner_scale,
        )
        @test collect(solutions.ours[1]) ≈ collect(solutions.reference[1]) rtol = 1.0f-3
        @test solutions.ours[2] ≈ solutions.reference[2] rtol = 1.0f-3
        @test solutions.ours[3:4] == solutions.reference[3:4]
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
    for (system, expected) in ((singular, SingularException), (non_finite, ArgumentError))
        @test solve_exception(GameTracing.matrix_bls(2, Float32), system) isa expected
        @test solve_exception(GameTracing.BifurcationKit.MatrixBLS(), system) isa expected
    end
end

@testset "a triangular bordered matrix takes the direct solve and agrees with the factored one" begin
    jacobian = Float32[2 0; 1 3]
    system = (;
        jacobian,
        border_column = Float32[0, 0],
        border_row = Float32[0, 0],
        corner = 5.0f0,
        residual = Float32[1, 2],
        border_value = 10.0f0,
    )
    solutions = bordered_solutions(system, 1.0f0, 1.0f0)
    @test collect(solutions.ours[1]) ≈ Float32[0.5f0, 0.5f0] rtol = 1.0f-5
    @test solutions.ours[2] ≈ 2.0f0 rtol = 1.0f-5
    @test solutions.ours[3:4] == solutions.reference[3:4]
    @test length(solutions.ours[1]) == 2
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
