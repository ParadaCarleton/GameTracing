using GameTracing: arclength_distance, corrected_accepted, logit_change, step_unresolved, largest_kl, max_regret, max_regret!, regret_workspace

"""The bytes a second call of `call` on `arguments` allocates, after a first call compiles it."""
function warm_bytes(call::F, arguments...)::Integer where {F}
    call(arguments...)
    return @allocated call(arguments...)
end

"""`gambit_settings`-like trace settings in `E` that allow `max_steps` steps."""
function step_budget_settings(::Type{E}, max_steps::Integer)::TraceSettings{E} where {E <: Real}
    return TraceSettings{E}(;
        first_step = E(0.03f0), aggressiveness = E(0.1f0), max_steps,
        geometry = TraceGeometry{E}(; distance = FisherLogPrecisionDistance),
    )
end

"""The bytes a warm `logit_solve` of `gains` at `maxregret` allocates under a step budget of `max_steps`."""
function solve_bytes(gains::NormalFormGains, ::Type{E}, max_steps::Integer, maxregret::Real)::Integer where {E <: Real}
    return warm_bytes(gains -> logit_solve(gains; maxregret, settings = step_budget_settings(E, max_steps)), gains)
end

@testset "logit_solve allocates by the points it records, whatever max_steps allows ($element)" for element in (Float32, widen(Float32))
    gains = NormalFormGains(map(table -> element.(table), LOGITNASH_GAMES.prisoners_dilemma))
    @test solve_bytes(gains, element, 20_000, 1.0f-3) <= solve_bytes(gains, element, 2_000, 1.0f-3) + 1024
end

@testset "a warm in-place bordered solve allocates nothing ($element)" for element in (Float32, widen(Float32))
    rng = fresh_rng()
    width = 4
    solver = GameTracing.bordered_linear_solver(GameTracing.MatrixSolve, width, element)
    jacobian = randn(rng, element, width, width) + element[row == column ? 4 : 0 for row in 1:width, column in 1:width]
    border_column, border_row, residual = randn(rng, element, width), randn(rng, element, width), randn(rng, element, width)
    @test warm_bytes(solver, jacobian, border_column, border_row, one(element), residual, one(element)) == 0
end

@testset "the regret check and the corrector's logit measures allocate nothing ($element)" for element in (Float32, widen(Float32))
    rng = fresh_rng()
    gains = NormalFormGains(ntuple(_ -> randn(rng, element, (3, 4)), 2))
    logits = randn(rng, element, logit_count(gains))
    work = regret_workspace(gains, logits)
    @test warm_bytes(max_regret!, work, gains, logits) == 0
    @test max_regret!(work, gains, logits) == max_regret(gains, logits)
    from, to = randn(rng, element, 6), randn(rng, element, 6)
    for measure in (logit_change, step_unresolved, largest_kl)
        @test warm_bytes(measure, from, to) == 0
    end
    @test warm_bytes(corrected_accepted, GameTracing.AcceptAll(), to, one(element), true) == 0
    for distance in instances(GameTracing.StepDistance)
        geometry = TraceGeometry{element}(; distance)
        displacement = to .- from
        @test warm_bytes(arclength_distance, from, to, geometry) == 0
        @test arclength_distance(from, to, geometry) ≈ sqrt(GameTracing.trace_dot(displacement, displacement, geometry))
    end
end
