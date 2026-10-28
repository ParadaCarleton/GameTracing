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
    sink::Channel{Tuple{String, Int, Float32, String}}
end
min_enabled_level(::Recorder)::typeof(Debug) = Debug
shouldlog(::Recorder, ::LogLevel, ::Module, group::Any, ::Any)::Bool = group in (:trace, :gambit)
catch_exceptions(::Recorder)::Bool = false
function handle_message(recorder::Recorder, ::LogLevel, message::Any, ::Module, ::Any, ::Any, ::Any, ::Any; keywords...)::Nothing
    measured = get(keywords, :distance, get(keywords, :step_length, NaN32))
    put!(recorder.sink, (string(message), Int(get(keywords, :step, -1)), Float32(measured), string(get(keywords, :reason, get(keywords, :converged, "")))))
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
    records = Channel{Tuple{String, Int, Float32, String}}(Inf)
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
        " max_newton=", maximum(newton; init = 0.0f0), " max_chord=", maximum(chords; init = 0.0f0), " profile=", repr(profile))
    if is_error(result)
        print_failure(arm, name, target, pass, unwrap_error(result).context, logged, newton)
    end
    if endswith(arm, "noguard")
        # Per-step chords in Gambit's coordinates with each step's starting λ, for a ratio over the λ range Gambit covers.
        println("CHORDS arm=", arm, " fixture=", name, " target=", target, " chords=", join([string(last(before), ":", chord) for (before, chord) in zip(starts, chords)], ","))
        opening = [string(record[1] == "accepted ds" ? "ds=" : string(record[1], ":", record[4], "@"), record[3]) for record in logged
                   if record[1] in ("accepted ds", "corrector closed", "corrector refused", "trace stopped")]
        println("OPENING arm=", arm, " fixture=", name, " target=", target, " first_step=", settings.first_step,
            " first_chords=", repr(chords[1:min(end, 6)]), " events=", join(opening[1:min(end, 16)], " "))
    end
    if occursin("battle_of_the_sexes", name) && endswith(arm, "noguard") && target in (20, 100)
        # Accepted points as [λ; p]: the step in ρ through Gambit's turn at λ ≈ 0.808.
        println("POINTS arm=", arm, " target=", target, " points=", repr([[last(point); exp.(point[1:(end - 1)])] for point in accepted]))
    end
    return nothing
end

"""Print why a trace stopped: refusal and stop reasons, its last step, and the Newton steps near its end against the coordinates' rounding."""
function print_failure(arm::AbstractString, name::AbstractString, target::Real, pass::P, context::NamedTuple, logged::AbstractVector, newton::AbstractVector)::Nothing where {P}
    reasons = [string(record[1], ":", record[4]) for record in logged if !isempty(record[4])]
    tally = Dict(reason => count(==(reason), reasons) for reason in unique(reasons))
    coordinates = [0.0f0]
    if haskey(context, :logits) && haskey(context, :relative_precision)
        coordinates = [GameTracing.log_probabilities(pass, collect(Float32, context.logits)); Float32(context.relative_precision * target)]
    end
    tail = newton[max(firstindex(newton), lastindex(newton) - 19):end]
    # A failure at the start carries fewer context fields than one past it.
    println("FAILINFO arm=", arm, " fixture=", name, " target=", target, " steps=", get(context, :steps, -1), " lambda=", get(context, :precision, NaN32),
        " step_length=", get(context, :step_length, NaN32), " unmoved=", get(context, :unmoved, -1), " reasons=", repr(tally),
        " tail_newton_min=", minimum(tail; init = Inf32), " coordinate_ulp=", eps(maximum(abs, coordinates)),
        " exception=", first(repr(get(context, :exception, nothing)), 160))
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
    # The symmetric games' pitchforks unfolded: player 1's (a1, a1) payoff raised by δ, so the branch is unique.
    perturbed = [(string(key, "_p", δ), (Float32.(LOGITNASH_GAMES[key][1]) .+ δ .* Float32[1 0; 0 0], Float32.(LOGITNASH_GAMES[key][2])))
                 for key in (:battle_of_the_sexes, :coordination_pure, :coordination_wiki) for δ in (1.0f-3, 1.0f-2)]
    foreach(fixture -> print_payoffs(fixture...), [fixtures; far; perturbed; [("unfolded", UNFOLDED_PITCHFORK_GAME)]])
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
        for target in (4.0f0, 20.0f0, 1.0f2), (name, payoffs) in perturbed
            run_trace(label, name, NormalFormPass(map(table -> Float32.(table), payoffs)), 2, target, arm(target))
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
    # Battle of the sexes in Float64: does precision decide the leg at its pitchfork (λ ≈ 0.808)?
    bos = NormalFormPass(map(table -> Float64.(table), LOGITNASH_GAMES.battle_of_the_sexes))
    for target in (20.0, 100.0)
        run_trace("baseline_noguard", "battle_of_the_sexes_f64", bos, 2, target, TraceSettings{Float64}())
        summed = TraceSettings{Float64}(; geometry = TraceGeometry{Float64}(; distance = SummedDistance),
            corrector = CorrectorSettings{Float64}(; newton_step_tolerance = 1.0e-4, newton_iterations = 100),
            step_control = StepControl{Float64}(; distance_limit = 0.4, contraction_limit = 0.6, growth_limit = 1.1, contraction_offset = 1.0, gambit = true),
            first_step = 0.03 / target)
        run_trace("summed_noguard", "battle_of_the_sexes_f64", bos, 2, target, summed)
    end
    widened = ThresholdPass(widen(FOLD_GAME.amplitude), widen(FOLD_GAME.threshold))
    println("FOLD_REFERENCE logits=", repr(branch_reference(widened, 1, float(2)).logits))
    return nothing
end
main()
