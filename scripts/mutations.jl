# Mutation check: each mutant breaks one line of src/ and the test suite must fail.
# Usage: julia --project scripts/mutations.jl [index ...]

const MUTANTS = [
    (
        "src/callbacks.jl",
        "check.accept_corrected(sinh.(state_of(iterate.x)), iterate.p, converged)",
        "check.accept_corrected(state_of(iterate.x), iterate.p, converged)",
    ),
    (
        "src/callbacks.jl",
        "finalise.keep_going(sinh.(state_of(point.u)), point.p)",
        "finalise.keep_going(state_of(point.u), point.p)",
    ),
    (
        "src/callbacks.jl",
        "check.reach * check.step_length",
        "check.reach / check.step_length",
    ),
    (
        "src/callbacks.jl",
        "progress.unmoved = point.p == progress.relative_precision &&",
        "progress.unmoved = point.p == progress.relative_precision ||",
    ),
    ("src/callbacks.jl", "if !(check.step_length > 0)", "if check.step_length > 0"),
    (
        "src/trace.jl",
        "isnothing(outcome.exception) && progress.relative_precision == target",
        "isnothing(outcome.exception)",
    ),
    (
        "src/trace.jl",
        "progress.steps >= step_limit && !progress.unmoved",
        "progress.steps > step_limit && !progress.unmoved",
    ),
    (
        "src/trace.jl",
        "settings.bordered_solver == MatrixSolve",
        "settings.bordered_solver != MatrixSolve",
    ),
    (
        "src/trace.jl",
        "target_precision * progress.relative_precision",
        "target_precision / progress.relative_precision",
    ),
    ("src/trace.jl", "sqrt(state_weight / width)", "sqrt(width / state_weight)"),
    (
        "src/branch.jl",
        "-pass.precision_slope[slot] * game.target_precision",
        "-pass.precision_slope[slot]",
    ),
    (
        "src/branch.jl",
        "((slot == rival) - pass.logit_slope[slot, rival])",
        "((slot == rival) + pass.logit_slope[slot, rival])",
    ),
    (
        "src/branch.jl",
        "residual[end] = relative_precision - pin",
        "residual[end] = relative_precision + pin",
    ),
    (
        "src/branch.jl",
        "augmented = θ * (count + 1) / count",
        "augmented = θ * count / (count + 1)",
    ),
    (
        "src/bordered.jl",
        "istril(bordered) || istriu(bordered)",
        "istril(bordered) && istriu(bordered)",
    ),
    ("src/failure.jl", "contains(message, \"LAPACK\")", "contains(message, \"XLAPACK\")"),
    (
        "src/failure.jl",
        "startswith(exception.msg, \"Newton failed to converge\")",
        "startswith(exception.msg, \"Newton did not converge\")",
    ),
]

"""Whether the test suite passes in `tree`."""
function suite_passes(tree::String)::Bool
    return success(pipeline(
        `julia --project=$tree -e "using Pkg; Pkg.test()"`;
        stdout = devnull,
        stderr = devnull,
    ))
end

function main(chosen::Vector{Int})::Nothing
    root = dirname(@__DIR__)
    survivors = String[]
    for index in chosen
        file, from, to = MUTANTS[index]
        tree = mktempdir()
        cp(root, tree; force = true)
        path = joinpath(tree, file)
        text = read(path, String)
        if !occursin(from, text)
            error("mutant $index: `$from` is not in $file")
        end
        write(path, replace(text, from => to; count = 1))
        caught = !suite_passes(tree)
        println(caught ? "caught   " : "SURVIVED ", index, " ", file, ": ", from)
        if !caught
            push!(survivors, "$index $file")
        end
        rm(tree; recursive = true)
    end
    println(length(survivors), " of ", length(chosen), " mutants survived")
    return nothing
end

main(isempty(ARGS) ? collect(eachindex(MUTANTS)) : parse.(Int, ARGS))
