using GameTracing
using BifurcationKit: BifurcationKit
using ErrorTypes: is_error, unwrap, unwrap_error
using Logging: AbstractLogger, Debug, LogLevel, with_logger
import Logging: catch_exceptions, handle_message, min_enabled_level, shouldlog
using Random: Xoshiro

include("games.jl")
include("fixtures.jl")

"""The seeded far-target games of `opening_tests.jl`: strategy counts `(3, 3)` and `(2, 2, 2)`, payoffs `3 randn`."""
function far_opening_games(rng::Xoshiro)::AbstractVector
    return [ntuple(_ -> 3 .* randn(rng, Float32, counts), length(counts)) for counts in ((3, 3), (2, 2, 2))]
end

# EXPERIMENT: Gambit's step control (path.cc) in every arm, against main's defaults; prints one RUN line per trace.

"""Debug records of a trace: corrector iterates and Gambit-coordinate Newton steps."""
struct Recorder <: AbstractLogger
    sink::Channel{Tuple{String, Int, Float32}}
end
min_enabled_level(::Recorder)::typeof(Debug) = Debug
shouldlog(::Recorder, ::LogLevel, ::Module, group::Any, ::Any)::Bool = group in (:trace, :gambit)
catch_exceptions(::Recorder)::Bool = false
function handle_message(recorder::Recorder, ::LogLevel, message::Any, ::Module, ::Any, ::Any, ::Any, ::Any; keywords...)::Nothing
    put!(recorder.sink, (string(message), Int(get(keywords, :step, -1)), Float32(get(keywords, :distance, NaN32))))
    return nothing
end

"""A keep_going that records each accepted point's Gambit coordinates into `sink`."""
struct ChordLog{P}
    pass::P
    target::Float32
    sink::Channel{Vector{Float32}}
end
function (chords::ChordLog)(logits::AbstractVector, relative_precision::Real)::Bool
    put!(chords.sink, [GameTracing.log_probabilities(chords.pass, collect(Float32, logits)); Float32(relative_precision) * chords.target])
    return true
end

"""The step guard's KL limit: 0.5 on, `nothing` off."""
function guard_limit(guard::Bool)::Union{Nothing, Float32}
    if guard
        return 0.5f0
    end
    return nothing
end

"""An arm's settings at `target`: Gambit's step control in `geometry` (or main's defaults where `geometry` is `nothing`)."""
struct Arm
    geometry::Union{Nothing, TraceGeometry{Float32}}
    guard::Bool
    max_logit_step::Union{Nothing, Float32}
end

function (arm::Arm)(target::E; corrector_keywords::NamedTuple = (;), overrides...)::TraceSettings{E} where {E <: Real}
    if isnothing(arm.geometry)
        corrector = CorrectorSettings{E}(; max_step_kl = guard_limit(arm.guard), corrector_keywords...)
        return TraceSettings{E}(; corrector, overrides...)
    end
    corrector = CorrectorSettings{E}(; max_step_kl = guard_limit(arm.guard), newton_step_tolerance = 1.0f-4, newton_iterations = 100,
        arm.max_logit_step, corrector_keywords...)
    step_control = StepControl{E}(; distance_limit = 0.4f0, contraction_limit = 0.6f0, growth_limit = 1.1f0, contraction_offset = 1.0f0, gambit = true)
    return TraceSettings{E}(; arm.geometry, corrector, step_control, first_step = 0.03f0 / target, overrides...)
end

const ARM_KINDS = (
    summed = (TraceGeometry{Float32}(; distance = SummedDistance), nothing),
    fisher = (TraceGeometry{Float32}(; distance = FisherDistance), nothing),
    fisher_dz1 = (TraceGeometry{Float32}(; distance = FisherDistance), 1.0f0),
    baseline = (nothing, nothing),
)

"""Print a fixture's payoff tables once, for the local pygambit comparison."""
function print_payoffs(name::AbstractString, payoffs::Tuple)::Nothing
    println("PAYOFFS ", name, " size=", repr(size(first(payoffs))), " tables=", repr([vec(Float32.(table)) for table in payoffs]))
    return nothing
end

"""Trace `pass` under `settings` and print its RUN line."""
function run_trace(arm::AbstractString, name::AbstractString, pass::P, width::Integer, target::E, settings::TraceSettings{E})::Nothing where {P, E <: Real}
    records = Channel{Tuple{String, Int, Float32}}(Inf)
    points = Channel{Vector{Float32}}(Inf)
    result = with_logger(() -> trace_endpoint(LogitGame(pass, target), width, settings; keep_going = ChordLog(pass, target, points)), Recorder(records))
    close(records)
    close(points)
    logged = collect(records)
    iterations = count(record -> record[1] == "corrector iterate" && record[2] >= 1, logged)
    newton = [record[3] for record in logged if record[1] == "newton step" && isfinite(record[3])]
    accepted = collect(points)
    starts = [[[GameTracing.log_probabilities(pass, zeros(Float32, width)); 0.0f0]]; accepted]
    chords = [sqrt(sum(abs2, after .- before)) for (before, after) in zip(starts, accepted)]
    outcome = is_error(result) ? string("FAIL:", unwrap_error(result).cause) : string(unwrap(result).steps)
    profile = is_error(result) ? Float32[] : exp.(GameTracing.log_probabilities(pass, unwrap(result).logits))
    println("RUN arm=", arm, " fixture=", name, " target=", target, " steps=", outcome, " iterations=", iterations,
        " max_newton=", maximum(newton; init = NaN32), " max_chord=", maximum(chords; init = NaN32), " profile=", repr(profile))
    return nothing
end

"""The arm benchmark's random game `seed` (`random_case`): the player count, then the tables, from one stream."""
function random_fixture(seed::Integer)::Tuple
    rng = Xoshiro(seed)
    players = rand(rng, 2:5)
    return random_payoffs(rng, players)
end

function main()::Nothing
    fixtures = vcat(
        [(string(key), payoffs) for (key, payoffs) in pairs(LOGITNASH_GAMES)],
        [("overshot", OVERSHOT_GAME)],
        [(string("random", seed), random_fixture(seed)) for seed in 1:24],
    )
    far = [(string("far", index), payoffs) for (index, payoffs) in enumerate(far_opening_games(Xoshiro(0xb0bf341028e24ba6)))]
    foreach(fixture -> print_payoffs(fixture...), [fixtures; far; [("unfolded", UNFOLDED_PITCHFORK_GAME)]])
    loose = (; residual_tolerance = 32 * eps(Float32) * 12)
    fold_variants = (
        fold_default = (;),
        fold_secant = (; tangent = BifurcationKit.Secant()),
        fold_bordering = (; bordered_solver = BorderingSolve),
        fold_loose = (; bordered_solver = BorderingSolve, corrector_keywords = loose),
    )
    for guard in (false, true), (kind, (geometry, max_logit_step)) in pairs(ARM_KINDS)
        arm = Arm(geometry, guard, max_logit_step)
        label = string(kind, guard ? "_guard" : "_noguard")
        for target in (1.0f2, 1.0f4, 1.0f6), (name, payoffs) in fixtures
            run_trace(label, name, NormalFormPass(map(table -> Float32.(table), payoffs)), length(payoffs), target, arm(target))
        end
        for target in (4.0f0, 20.0f0), (key, payoffs) in pairs(LOGITNASH_GAMES)
            if size(first(payoffs)) == (2, 2)
                run_trace(label, string(key), NormalFormPass(map(table -> Float32.(table), payoffs)), 2, target, arm(target))
            end
        end
        run_trace(label, "unfolded", NormalFormPass(map(table -> Float32.(table), UNFOLDED_PITCHFORK_GAME)), 2, 4.0f0, arm(4.0f0))
        for (name, payoffs) in far
            gains = NormalFormGains(payoffs)
            run_trace(label, name, gains, GameTracing.logit_count(gains), 1.0f6, arm(1.0f6))
        end
        for (variant, overrides) in pairs(fold_variants)
            run_trace(label, string(variant), FOLD_GAME, 1, 2.0f0, arm(2.0f0; overrides...))
        end
        if isnothing(geometry)
            run_trace(label, "fold_uncontrolled", FOLD_GAME, 1, 2.0f0, arm(2.0f0; step_control = nothing))
        end
    end
    widened = ThresholdPass(widen(FOLD_GAME.amplitude), widen(FOLD_GAME.threshold))
    println("FOLD_REFERENCE logits=", repr(branch_reference(widened, 1, float(2)).logits))
    return nothing
end
main()
