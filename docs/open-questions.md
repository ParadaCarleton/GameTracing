Questions for the owner. Answer in place; delete a line once settled.

- LICENSE: choose one for GameTracing and CodeRules.jl (both public, none set).
- `state_weight` default 0.5 throws for a one-logit trace. Derive it from the width or reject at construction?
- Mutation gate: `--max-sites 100` assumed from "cut it to 100". Confirm, then set thresholds from the first CI score.
- `CorrectorReach` and `TraceProgress` stay `mutable struct`. Removing them needs a PALC change in the BifurcationKit fork.
- `primal` export: drop it and export the `TraceCause` members?
- `TraceSettings` split: corrector controls moved into `CorrectorSettings` (15 fields to 12). Keep this grouping?
- `StepControl` fields renamed `distance_limit`, `contraction_limit`, `growth_limit`. Keep these names?
- CodeRules now exempts `Bool` results and types generic methods at called signatures. irv-extremism baselines need a re-ratchet.
