using BifurcationKit: BifurcationKit
using LinearAlgebra: LU, issuccess, istril, istriu, ldiv!, logabsdet, lu, lu!
"""How PALC solves its bordered linear system: `MatrixSolve` factors the whole bordered matrix, `BorderingSolve` the Jacobian alone."""
@enum BorderedSolver MatrixSolve BorderingSolve

"""
    InPlaceMatrixBLS(factorization, solver)

BifurcationKit's `MatrixBLS` with the bordered matrix written into `factorization`'s factors and LU-factored there
(`lu!`), without `MatrixBLS`'s `vcat`/`hcat` copies; its solutions are bit-identical. Any real element type factors:
LAPACK's for the BLAS floats, the generic LU otherwise. A triangular bordered matrix goes to `\\`. `solver` is never
read; it keeps PALC's `update` from replacing this solver with one built from the Newton options. The system is
`[J dR; ξu dzu' ξp dzp] [dX; dl] = [R; n]`, `J` the trace's `width × width` Jacobian.
"""
struct InPlaceMatrixBLS{E <: Real, S <: BifurcationKit.AbstractLinearSolver} <:
    BifurcationKit.AbstractBorderedLinearSolver
    factorization::LU{E, Matrix{E}, Vector{Int}}
    solver::S
end

function (bls::InPlaceMatrixBLS)(
        jacobian::AbstractMatrix,
        dR::AbstractVector,
        dzu::AbstractVector,
        dzp::T,
        residual::AbstractVector,
        border_value::T,
        ξu::T = one(T),
        ξp::T = one(T);
        dotp::Any = nothing,
        applyξu!::Any = nothing,
    )::Tuple where {T <: Number}
    bordered = bordered_matrix!(bls.factorization.factors, jacobian, dR, dzu, dzp, ξu, ξp, applyξu!)
    width = size(bordered, 1) - 1
    rhs = vcat(residual, border_value)
    if istril(bordered) || istriu(bordered)
        solution = bordered \ rhs
        return view(solution, 1:width), solution[end], true, 1
    end
    # `check` rejects non-finite entries and throws `SingularException` on a zero pivot, as `MatrixBLS`'s `lu` does.
    ldiv!(lu!(bls.factorization, bordered; check = true), rhs)
    return view(rhs, 1:width), rhs[end], true, 1
end

"""
    bordered_matrix!(bordered, jacobian, dR, dzu, dzp, ξu, ξp, applyξu!)

`bordered` filled with `MatrixBLS`'s bordered matrix `[jacobian dR; ξu dzu' ξp dzp]`, its border
row passed through `applyξu!` unless that is nothing.
"""
function bordered_matrix!(
        bordered::AbstractMatrix,
        jacobian::AbstractMatrix,
        dR::AbstractVector,
        dzu::AbstractVector,
        dzp::T,
        ξu::T,
        ξp::T,
        applyξu!::Any,
    )::typeof(bordered) where {T <: Number}
    width = size(jacobian, 1)
    if size(bordered, 1) != width + 1
        throw(
            DimensionMismatch(
                "a $(size(bordered, 1))-row bordered matrix for a $width-slot Jacobian",
            )
        )
    end
    copyto!(@view(bordered[1:width, 1:width]), jacobian)
    copyto!(@view(bordered[1:width, width + 1]), dR)
    for (column, tangent) in pairs(dzu)
        bordered[width + 1, column] = adjoint(tangent * ξu)
    end
    bordered[width + 1, width + 1] = dzp * ξp
    if !isnothing(applyξu!)
        applyξu!(@view(bordered[width + 1, 1:width]))
    end
    return bordered
end

"""
The `Bordered` tangent of PALC and the sign of the bordered matrix's determinant that its
`orientation_check` reads, from one LU: the matrix is built as `MatrixBLS` builds it, from PALC's `θ` and `dotθ`,
factored in `bls`'s buffers (`lu!`), the tangent solves with that factorization, and the sign is its determinant's
(`logabsdet`). A zero pivot makes the matrix singular: the sign and the tangent are zero, which PALC's NaN checks
refuse as a step.
"""
function BifurcationKit.bordered_tangent_and_sign(
        bls::InPlaceMatrixBLS,
        iter::BifurcationKit.AbstractContinuationIterable,
        state::BifurcationKit.AbstractContinuationState,
        jacobian::AbstractMatrix,
        dFdl::AbstractVector,
        rhsu::AbstractVector,
        rhsp::Number,
    )::Tuple
    state_weight = iter.alg.θ
    tangent = state.τ
    bordered = bordered_matrix!(
        bls.factorization.factors, jacobian, dFdl, tangent.u, tangent.p, state_weight, one(state_weight) - state_weight,
        iter.alg.dotθ.apply!,
    )
    factorization = lu!(bls.factorization, bordered; check = false)
    solution = vcat(rhsu, rhsp)
    if !issuccess(factorization)
        return view(zero(solution), 1:(length(solution) - 1)),
            zero(eltype(solution)),
            zero(eltype(solution))
    end
    ldiv!(factorization, solution)
    return view(solution, 1:(length(solution) - 1)), solution[end], last(logabsdet(factorization))
end

"""
    bordered_linear_solver(name, width, element)

PALC's bordered linear solver: `BorderingSolve` factors the Jacobian alone (`BorderingBLS`), `MatrixSolve`
factors the whole bordered matrix in place (`InPlaceMatrixBLS`), in any real element type.
"""
function bordered_linear_solver(
        solver::BorderedSolver,
        width::Integer,
        ::Type{E},
    )::BifurcationKit.AbstractBorderedLinearSolver where {E <: Real}
    if solver == BorderingSolve
        return BifurcationKit.BorderingBLS(BifurcationKit.DefaultLS())
    end
    # The identity factors without pivoting, so the buffers start valid for any element type.
    identity = Matrix{E}(undef, width + 1, width + 1)
    identity .= zero(E)
    for diagonal in axes(identity, 1)
        identity[diagonal, diagonal] = one(E)
    end
    return InPlaceMatrixBLS(lu(identity), BifurcationKit.DefaultLS())
end
