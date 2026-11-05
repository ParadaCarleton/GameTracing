# GameTracing

Homotopy continuation for logit quantal response equilibria. `trace_endpoint` follows a logit game's branch from
zero precision (infinite temperature), where the only equilibrium is the flat one, up to a target precision. It uses
BifurcationKit's pseudo-arclength continuation to cross folds.
