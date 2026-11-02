# The entry signatures Corset walks to type GameTracing (`[preferences.Corset] entry_points`), on two- and three-player games.

using GameTracing

# Corset includes this file in a bare `Module`, which binds no `include` of its own.
Base.include(@__MODULE__, "games.jl")
Base.include(@__MODULE__, "fixtures.jl")

"""The signatures of `GameTracing`'s solver entry points."""
function package_signatures()::Vector{Type}
    gains = [
        NormalFormGains(map(table -> Float32.(table), game))
        for game in (LOGITNASH_GAMES.matching_pennies, OVERSHOT_GAME)
    ]
    game = LogitGame(LinearGainPass(Float32[0 1; 1 0], Float32[0.5f0, -0.25f0]), 2.0f0)
    return Type[
        Tuple{typeof(trace_endpoint), typeof(game), Int, TraceSettings{Float32}},
        (Tuple{typeof(entry), typeof(gain)} for entry in (logit_solve, logit_solve_branch) for gain in gains)...,
        (Tuple{typeof(logit_solve_lambda), typeof(gain), Float32} for gain in gains)...,
    ]
end
