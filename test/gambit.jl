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
the trace's precision (the test "Gambit's λ is the trace's precision"). `precision` is the `λ` Gambit reports reaching.
"""
function gambit_branch(payoffs::Tuple, precision::Real)::NamedTuple
    gambit, numpy = pyimport("pygambit"), pyimport("numpy")
    game = gambit.Game.from_arrays((numpy.asarray(float.(table)) for table in payoffs)...)
    profile = first(gambit.qre.logit_solve_lambda(game; lam = [float(precision)]))
    first_choices = [pyconvert(Real, profile.profile[first(player.strategies)]) for player in game.players]
    return (; precision = pyconvert(Real, profile.lam), first_choices)
end
