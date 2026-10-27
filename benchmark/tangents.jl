# Counts, per perturbation scale and tangent, how often the trace of a perturbed battle of the sexes agrees with the
# arclength branch_reference to 1e-3: the measurement behind logitnash_tests' unfolded-pitchfork comment.
using GameTracing
using BifurcationKit: BifurcationKit
using ErrorTypes: is_error, unwrap
using LogitNash: LogitNash
using Random: Xoshiro, RandomDevice

include(joinpath(@__DIR__, "..", "test", "games.jl"))

const BATTLE_OF_THE_SEXES = ([3 0; 0 2], [2 0; 0 3])
const DRAWS = 40

function main()::Nothing
    seed = rand(RandomDevice(), UInt)
    println("SEED $seed")
    rng = Xoshiro(seed)
    for scale in inv.(float.((10^4, 10^3, 10^2, 10)))
        counts = Dict("bordered" => 0, "secant" => 0)
        crossed = 0
        for _ in 1:DRAWS
            payoffs = map(table -> table .+ scale .* randn(rng, 2, 2), map(float, BATTLE_OF_THE_SEXES))
            _, status = LogitNash.solve(payoffs; stop_lambda = 4.0, stop_eps = 0.0)
            precision = Float32(status.lambda)
            reference = branch_reference(NormalFormPass(payoffs), 2, precision)
            crossed += reference.crossings
            for (label, tangent) in ("bordered" => BifurcationKit.Bordered(), "secant" => BifurcationKit.Secant())
                game = LogitGame(NormalFormPass(map(table -> Float32.(table), payoffs)), precision)
                result = trace_endpoint(game, 2, TraceSettings{Float32}(; tangent))
                if !is_error(result) && reference.reached
                    traced = inv.(1 .+ exp.(-unwrap(result).logits))
                    counts[label] += maximum(abs, traced .- reference.first_choices) < 1.0e-3
                end
            end
        end
        println("SCALE $scale draws $DRAWS bordered $(counts["bordered"]) secant $(counts["secant"]) reference_crossings $crossed")
    end
    return nothing
end

main()
