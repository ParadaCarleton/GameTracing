using BifurcationKit: BifurcationKit
using LinearAlgebra: BlasFloat, BlasInt, LAPACK, LU, SingularException, istril, istriu, logabsdet
"""How PALC solves its bordered linear system: `MatrixSolve` factors the whole bordered matrix, `BorderingSolve` the Jacobian alone."""
@enum BorderedSolver MatrixSolve BorderingSolve

"""
    InPlaceMatrixBLS(matrix, pivots, solver)

BifurcationKit's `MatrixBLS` with the bordered matrix written into `matrix` and LU-factored there
with `pivots` (LAPACK `getrf`/`getrs`), without `MatrixBLS`'s `vcat`/`hcat` copies; its solutions
are bit-identical. A triangular bordered matrix goes to `\\`. `solver` is never read; it keeps PALC's
`update` from replacing this solver with one built from the Newton options. The system is
`[J dR; ξu dzu' ξp dzp] [dX; dl] = [R; n]`, `J` the trace's `width × width` Jacobian.
"""
struct InPlaceMatrixBLS{E <: BlasFloat, S <: BifurcationKit.AbstractLinearSolver} <:
    BifurcationKit.AbstractBorderedLinearSolver
    matrix::Matrix{E}
    pivots::Vector{BlasInt}
    solver::S
end

function (bls::InPlaceMatrixBLS)(
    jacobian::Any,
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
    bordered = bordered_matrix!(
        bls.matrix,
        BifurcationKit.getmatrix(jacobian),
        dR,
        dzu,
        dzp,
        ξu,
        ξp,
        applyξu!,
    )
    width = size(bordered, 1) - 1
    rhs = vcat(residual, border_value)
    if istril(bordered) || istriu(bordered)
        solution = bordered \ rhs
        return view(solution, 1:width), solution[end], true, 1
    end
    # `check` rejects only non-finite entries; a zero pivot shows in `info`, which `MatrixBLS`'s `lu` throws on.
    _, _, info = LAPACK.getrf!(bordered, bls.pivots; check = true)
    if info > 0
        throw(SingularException(info))
    end
    LAPACK.getrs!('N', bordered, bls.pivots, rhs)
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
        throw(DimensionMismatch(
            "a $(size(bordered, 1))-row bordered matrix for a $width-slot Jacobian",
        ))
    end
    # fatou-ignore call-arity: Base `view` takes any number of indices (jolars/fatou#125)
    copyto!(view(bordered, 1:width, 1:width), jacobian)
    # fatou-ignore call-arity: Base `view` takes any number of indices (jolars/fatou#125)
    copyto!(view(bordered, 1:width, width + 1), dR)
    for (column, tangent) in pairs(dzu)
        bordered[width + 1, column] = adjoint(tangent * ξu)
    end
    bordered[width + 1, width + 1] = dzp * ξp
    if !isnothing(applyξu!)
        # fatou-ignore call-arity: Base `view` takes any number of indices (jolars/fatou#125)
        applyξu!(view(bordered, width + 1, 1:width))
    end
    return bordered
end

"""
The sign of the bordered tangent matrix's determinant that PALC's `orientation_check` reads, taken
from the same LU as the solve (`logabsdet`, LAPACK `getrf` without a singularity check).
"""
function BifurcationKit.bordered_determinant_sign(
    bls::InPlaceMatrixBLS,
    iter::BifurcationKit.AbstractContinuationIterable,
    state::BifurcationKit.AbstractContinuationState,
    jacobian::Any,
    dFdl::AbstractVector,
)::eltype(dFdl)
    state_weight = BifurcationKit.getθ(iter)
    tangent = state.τ
    bordered = bordered_matrix!(
        bls.matrix,
        BifurcationKit.getmatrix(jacobian),
        dFdl,
        tangent.u,
        tangent.p,
        state_weight,
        one(state_weight) - state_weight,
        BifurcationKit._get_apply_dot(BifurcationKit.getdot(iter)),
    )
    _, _, info = LAPACK.getrf!(bordered, bls.pivots; check = false)
    return last(logabsdet(LU(bordered, bls.pivots, info)))
end

"""
    bordered_linear_solver(name, width, element)

PALC's bordered linear solver: `BorderingSolve` factors the Jacobian alone (`BorderingBLS`), `MatrixSolve`
factors the whole bordered matrix (`InPlaceMatrixBLS`, or BifurcationKit's `MatrixBLS` for an element type
LAPACK does not factor).
"""
function bordered_linear_solver(
    solver::BorderedSolver,
    width::Integer,
    ::Type{E},
)::BifurcationKit.AbstractBorderedLinearSolver where {E <: Real}
    if solver == BorderingSolve
        return BifurcationKit.BorderingBLS(BifurcationKit.DefaultLS())
    end
    return matrix_bls(width, E)
end

function matrix_bls(
    width::Integer,
    ::Type{E},
)::BifurcationKit.AbstractBorderedLinearSolver where {E <: BlasFloat}
    return InPlaceMatrixBLS(
        Matrix{E}(undef, width + 1, width + 1),
        Vector{BlasInt}(undef, width + 1),
        BifurcationKit.DefaultLS(),
    )
end

function matrix_bls(::Integer, ::Type{<:Real})::BifurcationKit.AbstractBorderedLinearSolver
    return BifurcationKit.MatrixBLS()
end
