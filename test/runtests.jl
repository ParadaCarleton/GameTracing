# EXPERIMENT: locate the random two-action game where the trace and Gambit disagree; not for landing.
using GameTracing
using Test
using Random: Xoshiro

include("games.jl")
include("fixtures.jl")
include("arms.jl")
include("gambit.jl")

const GAME = LinearGainPass(Float32[0 1; 1 0], Float32[0.5f0, -0.25f0])
const TARGET_PRECISION = 2.0f0
const WIDTH = 2

@testset "GameTracing" begin
    include("logitnash_tests.jl")
end

# The fresh seeds logged before the failure in job 123176229948, last first.
const SEEDS = (0x3b1c8fe648f6584a, 0x5a2113b075cefd41, 0x8372d5d2477bc8cb, 0xdfda3d8bbd9aeb19, 0x6b95bb220edfe07c,
    0x951dbfb24ef0f267, 0xbe7e952871064590)

"""Print one PROBE line per game of the two-action testset drawn from `seed`: each side's first choices."""
function probe(seed::Unsigned)::Nothing
    rng = Xoshiro(seed)
    for draw in 1:20
        players = rand(rng, 2:4)
        payoffs = ntuple(_ -> 3 .* randn(rng, ntuple(_ -> 2, players)), players)
        cross = logitnash_equilibrium(payoffs, 3)
        result = traced_result(payoffs, cross.precision, TraceSettings{Float32}())
        traced = is_error(result) ? Float32[] : first_choices(unwrap(result).logits)
        wide = trace_endpoint(LogitGame(NormalFormPass(map(float, payoffs)), Float64(cross.precision)), players, TraceSettings{Float64}())
        traced64 = is_error(wide) ? Float64[] : first_choices(unwrap(wide).logits)
        oracle = gambit_branch(payoffs, cross.precision)
        reference = branch_reference(NormalFormPass(map(float, payoffs)), players, cross.precision)
        flagged = isempty(traced) || !agreeing(traced, oracle.first_choices)
        println("PROBE seed=", repr(seed), " draw=", draw, " flagged=", flagged, " lambda=", cross.precision,
            " traced=", traced, " traced64=", traced64, " gambit=", oracle.first_choices, " logitnash=", cross.first_choices,
            " reference=", reference.first_choices, " reached=", reference.reached, " crossings=", reference.crossings)
        if flagged
            println("PAYOFFS seed=", repr(seed), " draw=", draw, " ", repr(payoffs))
        end
    end
    return nothing
end

foreach(probe, SEEDS)
