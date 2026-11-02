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

# Three players, two actions each (a random draw, rounded to 1e-6). Its principal branch folds at precision about 3.3617.
# Gambit's default steps round that fold in one step, from 3.349675 to 3.128675, so `logit_solve_lambda` misses the first
# crossing of any precision between them and returns a later one, near the pure profile (1, 2, 2).
const FOLDED_GAMBIT_GAME = (
    [-4108184 -6199065; -140340 -5617899;;; 493257 3009713; 540823 127869] .// 10^6,
    [-359879 -9625560; 3795891 -3394274;;; -3946851 1662104; 3214943 2863915] .// 10^6,
    [-503395 -5503689; -4888492 2357340;;; -1217139 -3504025; -3129362 207368] .// 10^6,
)

# Four players, two actions each (seed 0x2c06c04a5d81b3dc of the random-game testset, rounded to 1e-6). At precision
# about 3.05, Gambit leaves its principal branch, ending at first choices (0.16, 0.36, 0.08, 0.14) (CI run 41370200196).
const GAMBIT_LEAVES_GAME = (
    [1951724 1176029; 1415337 2765348;;; 998115 1201523; 2814368 -7103004;;;; -1878885 1184060; 5947423 -6109252;;; 1673502 -1027796; -888519 2955579] .// 10^6,
    [-2486602 566455; -2904504 -3764541;;; -5818249 30625; 2485436 618336;;;; 2033055 342635; -2769907 -1540549;;; -1633480 -2665809; 125757 659713] .// 10^6,
    [-2096866 1302905; 999183 2776673;;; -569484 -3032630; -4382120 7834803;;;; 1660815 -9421362; 2953784 3122954;;; -1815115 4488691; 2488710 2540069] .// 10^6,
    [-207281 1181961; 4483016 2010697;;; 295707 -1804204; 3555830 2867891;;;; 2785242 7372728; 2213823 994217;;; -3226846 3021391; 573187 5247358] .// 10^6,
)

# The battle of the sexes perturbed by about 1e-4. Near precision 0.808 its principal branch turns steeply (dλ/ds about
# 0.0024) past the middle arm of the unfolded pitchfork, and ends near the pure equilibrium (1, 1).
const UNFOLDED_PITCHFORK_GAME = ([30000726 -37; 757 20000821] .// 10^7, [19999477 -283; 919 29999523] .// 10^7)

# One logit whose branch folds back at precision about 1.4; target 2 lies past the fold.
const FOLD_GAME = ThresholdPass(6.0f0, 3.0f0)
