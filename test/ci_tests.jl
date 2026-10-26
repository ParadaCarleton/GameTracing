using YAML: YAML

# Source files the mutation job skips: the module file holds only includes and exports.
const UNMUTATED_SOURCES = ("GameTracing.jl",)

@testset "the mutation job mutates every source file but the listed exclusions" begin
    workflow = YAML.load_file(joinpath(pkgdir(GameTracing), ".github", "workflows", "CI.yml"))
    mutated = workflow["jobs"]["mutation"]["strategy"]["matrix"]["file"]
    sources = filter(endswith(".jl"), readdir(joinpath(pkgdir(GameTracing), "src")))
    @test sort(mutated) == sort(setdiff(sources, UNMUTATED_SOURCES))
end
