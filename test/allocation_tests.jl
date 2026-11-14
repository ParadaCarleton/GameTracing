using GameTracing: logit_change, step_unresolved, largest_kl, max_regret, max_regret!, regret_workspace

"""The bytes a second call of `f` on `arguments` allocates, after a first call compiles it."""
function warm_bytes(f::F, arguments...)::Integer where {F}
    f(arguments...)
    return @allocated f(arguments...)
end

"""`gambit_settings`-like trace settings in `E` that allow `max_steps` steps."""
function step_budget_settings(::Type{E}, max_steps::Integer)::TraceSettings{E} where {E <: Real}
    return TraceSettings{E}(;
        first_step = E(0.03), aggressiveness = E(0.1), max_steps,
        geometry = TraceGeometry{E}(; distance = FisherLogPrecisionDistance),
    )
end

"""The bytes a warm `logit_solve` of `gains` at `maxregret` allocates under a step budget of `max_steps`."""
function solve_bytes(gains::NormalFormGains, ::Type{E}, max_steps::Integer, maxregret::Real)::Integer where {E <: Real}
    return warm_bytes(gains -> logit_solve(gains; maxregret, settings = step_budget_settings(E, max_steps)), gains)
end

@testset "logit_solve allocates by the points it records, whatever max_steps allows" begin
    for E in (Float32, Float64)
        gains = NormalFormGains(map(table -> E.(table), LOGITNASH_GAMES.prisoners_dilemma))
        @test solve_bytes(gains, E, 20_000, 1.0e-3) <= solve_bytes(gains, E, 2_000, 1.0e-3) + 1024
    end
end

@testset "the regret check and the corrector's logit measures allocate nothing" begin
    rng = fresh_rng()
    for E in (Float32, Float64)
        gains = NormalFormGains(ntuple(_ -> randn(rng, E, (3, 4)), 2))
        logits = randn(rng, E, logit_count(gains))
        work = regret_workspace(gains, logits)
        @test warm_bytes(max_regret!, work, gains, logits) == 0
        @test max_regret!(work, gains, logits) == max_regret(gains, logits)
        from, to = randn(rng, E, 6), randn(rng, E, 6)
        for measure in (logit_change, step_unresolved, largest_kl)
            @test warm_bytes(measure, from, to) == 0
        end
    end
end
