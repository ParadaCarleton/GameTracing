# Games the tests and the arm benchmark share.

# LogitNash.jl's games with two actions per player (test/test_e2e.jl and its README), as integer payoff tables.
const LOGITNASH_GAMES = (
    battle_of_the_sexes = ([3 0; 0 2], [2 0; 0 3]),
    coordination_wiki = ([2 1; 1 2], [4 3; 3 4]),
    coordination_pure = ([8 0; 0 8], [8 0; 0 8]),
    coordination_assurance = ([8 0; 0 5], [8 0; 0 5]),
    stag_hunt = ([8 0; 7 5], [8 7; 0 5]),
    prisoners_dilemma = ([5 0; 8 1], [5 8; 0 1]),
    matching_pennies = ([1 -1; -1 1], [-1 1; 1 -1]),
    three_prisoners = (
        cat([3 1; 5 2], [1 0; 2 1]; dims = 3),
        cat([3 5; 1 2], [1 2; 0 1]; dims = 3),
        cat([3 1; 1 0], [5 2; 2 1]; dims = 3),
    ),
)

# Games symmetric under relabeling actions (battle of the sexes also swaps players): the principal branch meets an
# exact pitchfork (battle of the sexes at precision 0.808), past which it is ambiguous. Exact symmetry is measure
# zero, so these games assert only that the endpoint is an equilibrium of the map.
const PITCHFORK_GAMES = (:battle_of_the_sexes, :coordination_wiki, :coordination_pure)

# Three players, two actions each. The principal branch folds back in precision between 31//10 and 32//10, so a
# corrector started from a predictor past the target can converge onto the folded-back segment.
const OVERSHOT_GAME = (
    [358 -73; -175 -10;;; 113 -822; -685 390] .// 100,
    [183 -129; 53 -357;;; -158 235; -354 -285] .// 100,
    [-175 -405; 87 -124;;; 10 -177; -110 180] .// 100,
)

# One logit whose branch folds back at precision about 1.4; target 2 lies past the fold.
const FOLD_GAME = ThresholdPass(6.0f0, 3.0f0)

# The swap moves a two-player branch by under a tenth: a redraw a projection and a fixed-level correction absorb.
const NEAR_SWAP = MapSwap(
    LinearGainPass(Float32[0 0.8f0; 0.6f0 0], Float32[0.3f0, -0.2f0]),
    LinearGainPass(Float32[0 0.8f0; 0.6f0 0], Float32[0.34f0, -0.23f0]),
    4.0f0, 0.5f0, 2,
)

# The swap removes the low branch at precision 1/2: the only root there is the high one, about 3.5 logits away.
const FAR_SWAP = MapSwap(ThresholdPass(8.0f0, 4.0f0), ThresholdPass(8.0f0, 1.5f0), 1.0f0, 0.5f0, 1)
