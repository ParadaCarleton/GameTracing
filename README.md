# GameTracing

Homotopy continuation for logit quantal response equilibria. `trace_endpoint` follows a logit game's branch from
zero precision (infinite temperature), where the only equilibrium is the flat one, up to a target precision. It uses
BifurcationKit's pseudo-arclength continuation and crosses folds. Failures are values (`TraceFailure`).

Extracted from irv-extremism `src/consolidation.jl` (QRE trace) and `src/rootfinder.jl` at commit `a093a5d8`.
The code needs the `irv` branch of ParadaCarleton/BifurcationKit.jl (`[sources]`).
