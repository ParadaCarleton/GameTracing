# Replays the unfolded-pitchfork draws of one seed under each setting, and prints where a trace leaves the
# Float64 reference branch: the step, its chord in (w, ρ), the chord's cosine with the one before, and the endpoints.
using GameTracing
using ErrorTypes: is_error, unwrap, unwrap_error
using LinearAlgebra: dot, norm
using LogitNash: LogitNash
using Printf: @printf
using Random: Xoshiro

const TEST_DIRECTORY = joinpath(@__DIR__, "..", "test")
include(joinpath(TEST_DIRECTORY, "games.jl"))
const ARMS_FILE = joinpath(TEST_DIRECTORY, "arms.jl")
if isfile(ARMS_FILE)
    include(ARMS_FILE)
end

const SEED = 0x2c1ff8138445e8b0
# The fixtures' battle of the sexes; main keeps it inside logitnash_tests.jl, which runs testsets on include.
const BATTLE_OF_THE_SEXES = ([3 0; 0 2], [2 0; 0 3])
const POINT_CAPACITY = 4096

"""Accepted points `[logits; ρ]` of a trace, one per column, unfilled columns NaN: a `keep_going` that never stops."""
struct PathRecorder{M <: AbstractMatrix}
    points::M
end

function (recorder::PathRecorder)(logits::AbstractVector{E}, relative_precision::E)::Bool where {E <: Real}
    column = findfirst(isnan, view(recorder.points, 1, :))
    if column !== nothing
        recorder.points[:, column] .= [logits; relative_precision]
    end
    return true
end

"""The draws of the test "a pitchfork unfolded by a payoff perturbation" from `seed`, in its order."""
function unfolded_draws(seed::Unsigned)::Vector
    rng = Xoshiro(seed)
    base = map(float, BATTLE_OF_THE_SEXES)
    return [map(table -> table .+ scale .* randn(rng, 2, 2), base) for scale in inv.(float.((10^4, 10^3, 10^2, 10))) for _ in 1:3]
end

"""LogitNash's stopping precision past `precision` for `payoffs`, as the cross-check traces to it."""
function logitnash_precision(payoffs::Tuple, precision::Real)::Float32
    _, status = LogitNash.solve(map(float, payoffs); stop_lambda = float(precision), stop_eps = zero(float(precision)))
    return Float32(status.lambda)
end

"""The settings to replay: the defaults, and each trace-geometry arm where this tree has them."""
function replay_settings()::Vector{Pair{String, TraceSettings{Float32}}}
    defaults = ["default" => TraceSettings{Float32}()]
    if isdefined(Main, :arm_variants)
        return [defaults; Base.invokelatest(arm_variants)]
    end
    return defaults
end

"""`[w; ρ]` of an accepted point `[logits; ρ]`; the origin maps to itself."""
function gap_point(point::AbstractVector{E})::Vector{Float64} where {E <: Real}
    precision = Float64(last(point))
    return [Float64.(point[1:(end - 1)]) ./ max(precision, eps()); precision]
end

"""Print the first steps, the first accepted point off the reference branch, and the last point of `points`."""
function report_path(label::AbstractString, draw::Integer, pass::Any, points::AbstractMatrix, precision::Real)::Nothing
    width = size(points, 1) - 1
    previous = zeros(width + 1)
    previous_chord = fill(NaN, width + 1)
    left = false
    filled = count(!isnan, view(points, 1, :))
    for (step, point) in enumerate(eachcol(view(points, :, 1:filled)))
        here = gap_point(point)
        reference = branch_reference(pass, width, last(here) * precision)
        traced = inv.(1 .+ exp.(-Float64.(point[1:width])))
        gap = maximum(abs, traced .- reference.first_choices)
        chord = here .- previous
        cosine = dot(chord, previous_chord) / (norm(chord) * norm(previous_chord))
        flag = !left && gap > 1.0e-3
        if flag || step <= 3 || step == filled
            @printf("PATH %s draw %d step %d ρ %.5f Δρ %.5f |Δ(w,ρ)| %.5f cos %.4f probs %s ref %s gap %.5f%s\n",
                label, draw, step, last(here), last(chord), norm(chord), cosine,
                string(round.(traced; digits = 5)), string(round.(reference.first_choices; digits = 5)), gap, flag ? " LEFT" : "")
        end
        left = left || flag
        previous, previous_chord = here, chord
    end
    return nothing
end

function main()::Nothing
    for (draw, payoffs) in enumerate(unfolded_draws(SEED))
        precision = logitnash_precision(payoffs, 4)
        pass64 = NormalFormPass(map(float, payoffs))
        reference = branch_reference(pass64, 2, precision)
        for (label, settings) in replay_settings()
            game = LogitGame(NormalFormPass(map(table -> Float32.(table), payoffs)), precision)
            recorder = PathRecorder(fill(NaN32, 3, POINT_CAPACITY))
            result = trace_endpoint(game, 2, settings; keep_going = recorder)
            if is_error(result)
                @printf("RESULT %s draw %d failed %s\n", label, draw, string(unwrap_error(result).cause))
                continue
            end
            traced = inv.(1 .+ exp.(-unwrap(result).logits))
            gap = maximum(abs, traced .- reference.first_choices)
            jumped = reference.reached && gap > 1.0e-3
            @printf("RESULT %s draw %d precision %.4f steps %d probs %s ref %s jumped %s\n", label, draw, precision,
                unwrap(result).steps, string(round.(traced; digits = 5)), string(round.(reference.first_choices; digits = 5)), jumped)
            if jumped
                println("PAYOFFS draw $draw ", repr(payoffs))
                report_path(label, draw, pass64, recorder.points, precision)
            end
        end
    end
    return nothing
end

main()
