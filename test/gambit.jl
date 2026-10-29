using CondaPkg: CondaPkg

# Gambit's logit QRE is the test oracle, at the version checked by hand against a 50-digit trace.
# The pin goes into the test environment, not a package-level CondaPkg.toml that every user of GameTracing would read.
const PYGAMBIT_VERSION = "16.7.0"
CondaPkg.add("python"; version = "=3.12", resolve = false)
CondaPkg.add_pip("pygambit"; version = "==$PYGAMBIT_VERSION")
# PythonCall resolves the environment and starts Python when it loads, so it loads after the pins.
using PythonCall: pyconvert, pyimport

"""
    gambit_branch(payoffs, precision)

`(; precision, first_choices)` of Gambit's logit QRE branch of the two-action game `payoffs` (one table per player, axis
`k` player `k`'s action, first action first), from `pygambit.qre.logit_solve_lambda` at `precision`. Gambit's
`π ∝ exp(λ u)` on the raw payoffs is the logit map `z = λ G` with `G` the first action's payoff gain, so Gambit's `λ` is
the trace's precision (the test "Gambit's λ is the trace's precision"). `precision` is the `λ` Gambit reports reaching; both come back in `precision`'s type, the compared trace's.
"""
function gambit_branch(payoffs::Tuple, precision::E)::NamedTuple{(:precision, :first_choices), Tuple{E, Vector{E}}} where {E <: Real}
    gambit, numpy = pyimport("pygambit"), pyimport("numpy")
    game = gambit.Game.from_arrays((numpy.asarray(float.(table)) for table in payoffs)...)
    profile = first(gambit.qre.logit_solve_lambda(game; lam = [float(precision)]))
    first_choices = [pyconvert(E, profile.profile[first(player.strategies)]) for player in game.players]
    return (; precision = pyconvert(E, profile.lam), first_choices)
end

"""
Whether Gambit's logit branch of `payoffs` folds back in `λ` before it first reaches `precision`: a point of
`pygambit.qre.logit_solve_branch` below the point before it. `logit_solve_lambda` takes the same steps and finds a
crossing only between consecutive points, so a step that rounds such a fold can hide the first crossing.
"""
function gambit_folds_before(payoffs::Tuple, precision::Real)::Bool
    gambit, numpy = pyimport("pygambit"), pyimport("numpy")
    game = gambit.Game.from_arrays((numpy.asarray(float.(table)) for table in payoffs)...)
    # Regret 1e-12 of the payoff range ends the branch far past any precision these tests ask for.
    branch = gambit.qre.logit_solve_branch(game; maxregret = inv(float(10^12)))
    previous = zero(float(precision))
    for point in branch
        lam = pyconvert(typeof(float(precision)), point.lam)
        if lam >= precision
            return false
        end
        if lam < previous
            return true
        end
        previous = lam
    end
    return false
end

"""
Gambit's logit QRE of the normal-form game `payoffs` at `precision`, from `pygambit.qre.logit_solve_lambda`:
`(; precision, strategies)`, each player's mixed strategy in `precision`'s type.
"""
function gambit_strategies(payoffs::Tuple, precision::E)::NamedTuple where {E <: Real}
    gambit, numpy = pyimport("pygambit"), pyimport("numpy")
    game = gambit.Game.from_arrays((numpy.asarray(float.(table)) for table in payoffs)...)
    profile = first(gambit.qre.logit_solve_lambda(game; lam = [float(precision)]))
    strategies = Tuple([pyconvert(E, profile.profile[strategy]) for strategy in player.strategies] for player in game.players)
    return (; precision = pyconvert(E, profile.lam), strategies)
end
