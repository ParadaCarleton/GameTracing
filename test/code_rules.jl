# Corset's configured rules (`[preferences.Corset]` in Project.toml) on GameTracing and on the arm benchmark.

using Corset: Corset, Finding, entry_frames, load_config, run_rules, seed_sources, seeded_frames
using Corset.Entries: entry_signatures

const ROOT = pkgdir(GameTracing)
const CONFIG = load_config(ROOT)

@testset "code rules find nothing" begin
    frames = seeded_frames(seed_sources(CONFIG, ROOT, nothing, GameTracing, devnull), GameTracing, devnull)
    findings, skipped = run_rules(CONFIG, ROOT; mod = GameTracing, signatures = frames)
    @test findings == Finding[]
    @test skipped == String[]
end

"""The arm benchmark as a module, for the module rules; its script entry point does not run on load."""
module ArmBenchmark
include(joinpath(@__DIR__, "..", "benchmark", "arms.jl"))
end

@testset "the arm benchmark's rules find nothing" begin
    # The quick scope and one pitchfork draw call every function the full runs do; the rules type what they compiled.
    ArmBenchmark.main(joinpath(mktempdir(), "arms.csv"), "quick", devnull)
    ArmBenchmark.pitchfork_benchmark(joinpath(mktempdir(), "pitchfork.csv"), devnull, 1, ArmBenchmark.UNFOLDING_SEED)
    frames = entry_frames(entry_signatures(ArmBenchmark), ArmBenchmark)
    findings, skipped = run_rules(CONFIG, ROOT; mod = ArmBenchmark, signatures = frames)
    @test findings == Finding[]
    @test skipped == String[]
end
