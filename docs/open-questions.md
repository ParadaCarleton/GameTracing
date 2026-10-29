Questions for the owner. Answer in place; delete a line once settled.

- LICENSE: choose one for GameTracing and CodeRules.jl (both public, none set).
- `state_weight` default 0.5 throws for a one-logit trace. Derive it from the width or reject at construction?
- `CorrectorReach` and `TraceProgress` stay `mutable struct`. Removing them needs a PALC change in the BifurcationKit fork.
- Export the `TraceCause` members (`NoConvergence`, `TraceStuck`, `NotFinite`, `Singular`)?
- `TraceSettings` split: corrector controls moved into `CorrectorSettings` (15 fields to 12). Keep this grouping?
- `StepControl` fields renamed `distance_limit`, `growth_limit`; `contraction_limit` moved to `CorrectorSettings`. Keep these names?
- CodeRules now exempts `Bool` results and types generic methods at called signatures. irv-extremism baselines need a re-ratchet.
