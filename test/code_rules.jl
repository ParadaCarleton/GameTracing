# Corset on GameTracing with no baseline: every rule must find nothing.

using Corset:
    Repository,
    Site,
    abstract_fields,
    abstract_locals,
    boxed_closures,
    capturing_closures,
    concrete_arguments,
    concrete_returns,
    fatou_sites,
    float64_sites,
    hidden_inputs,
    inferred_methods,
    located_source,
    misplaced_imports,
    missing_annotations,
    module_structs,
    nothing_returns,
    one_letter_sites,
    range_guards,
    repeated_prefix,
    repeated_statements,
    repository_sources,
    return_type_calls,
    struct_locations,
    struct_site,
    unmarked_effects,
    where_return_sites

const STRUCT_FIELD_LIMIT = 12
const FILE_LINE_LIMIT = 1500
const SOURCE_DIRECTORIES = ("src", "test")

const REPOSITORY = Repository(dirname(@__DIR__); baseline = joinpath(@__DIR__, "no_baseline.toml"))
const SOURCE_FILES = [
    file
    for file in readdir(joinpath(pkgdir(GameTracing), "src"); join = true)
    if endswith(file, ".jl")
]

@testset "code rules find nothing" begin
    sources = Dict(file => located_source(file) for file in SOURCE_FILES)
    inferred = inferred_methods(GameTracing)
    structs = module_structs(GameTracing)
    locations = struct_locations(SOURCE_FILES)
    @testset "a function reads only its arguments and immutable constants" begin
        @test hidden_inputs(GameTracing) == Site[]
    end
    @testset "every local variable infers a concrete type" begin
        @test abstract_locals(REPOSITORY, inferred) == Site[]
    end
    @testset "a function that always returns nothing is named with a bang" begin
        @test unmarked_effects(REPOSITORY, inferred) == Site[]
    end
    @testset "no closure captures data or rebinds a captured variable" begin
        @test capturing_closures(GameTracing) == Site[]
        @test boxed_closures(GameTracing) == Site[]
    end
    @testset "structs hold at most $STRUCT_FIELD_LIMIT fields, each concretely typed, none repeating a prefix" begin
        @test [
            struct_site(nameof(type), locations)
            for type in structs
            if fieldcount(type) > STRUCT_FIELD_LIMIT
        ] ==
            Site[]
        @test [
            struct_site(nameof(type), locations)
            for type in structs
            if !isempty(abstract_fields(type))
        ] ==
            Site[]
        @test [
            struct_site(nameof(type), locations)
            for type in structs
            if repeated_prefix(type) !== nothing
        ] ==
            Site[]
    end
    @testset "no result-type guessing, no file over $FILE_LINE_LIMIT lines" begin
        @test [
            site
            for (file, located) in sources
            for site in return_type_calls(file, located)
        ] ==
            Site[]
        @test [file for file in SOURCE_FILES if countlines(file) > FILE_LINE_LIMIT] ==
            String[]
    end
    @testset "every quantity computed for its whole domain" begin
        @test [
            site
            for (file, located) in sources
            for site in nothing_returns(file, located)
        ] ==
            Site[]
        @test [
            site
            for (file, located) in sources
            for site in range_guards(file, located)
        ] ==
            Site[]
    end
    @testset "no statement repeats an earlier one in the same block" begin
        @test [
            site
            for (file, located) in sources
            for site in repeated_statements(file, located)
        ] ==
            Site[]
    end
    @testset "no argument is annotated with a concrete container" begin
        @test [
            site
            for (file, located) in sources
            for site in concrete_arguments(file, located)
        ] ==
            Site[]
    end
    @testset "no result is annotated with a concrete type unless it is generic" begin
        @test [
            site
            for (file, located) in sources
            for site in concrete_returns(file, located, GameTracing)
        ] ==
            Site[]
    end
    @testset "every function argument and result is annotated" begin
        @test [
            site
            for (file, located) in sources
            for site in missing_annotations(file, located)
        ] ==
            Site[]
    end
    checked = repository_sources(REPOSITORY, SOURCE_DIRECTORIES)
    @testset "no Float64 where Float32 is meant" begin
        @test [site for file in checked for site in float64_sites(file)] == Site[]
    end
    @testset "a short-form definition's `where` binds the method" begin
        @test [site for file in checked for site in where_return_sites(file)] == Site[]
    end
    @testset "no one-letter variable names" begin
        @test [site for file in checked for site in one_letter_sites(REPOSITORY, file)] ==
            Site[]
    end
    @testset "Fatou finds nothing" begin
        @test fatou_sites(REPOSITORY, SOURCE_DIRECTORIES) == Site[]
    end
    @testset "every import sits in the files that use it" begin
        @test misplaced_imports(REPOSITORY, SOURCE_DIRECTORIES) == Site[]
    end
end
