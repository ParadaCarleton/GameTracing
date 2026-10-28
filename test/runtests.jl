using GameTracing
using Test
using ErrorTypes: is_error, unwrap_error

include("games.jl")
include("gambit.jl")

# SCRATCH: the (2, 2, 2) game of seed 0x233ce38d667063a1 stalls at λ ≈ 22.9; log every trace event.
const BITS = (
    UInt32[0x40e91772 0x404e8c34; 0xc003afdd 0x408080df;;; 0x401102e2 0xc0c585fc; 0xc03dc62b 0x3f20f00a],
    UInt32[0xc00ce6e1 0xbf3fb89a; 0x3f8d6714 0xbf7847a8;;; 0x40401810 0x4084ab0e; 0xbf59c7b4 0xbfb6711c],
    UInt32[0xbf7283f2 0x3fbb443a; 0xc0235813 0xc02ac946;;; 0xbee03fc7 0x408a37b5; 0xc0cc44b0 0xc081998c],
)
function main()::Nothing
    gains = NormalFormGains(map(table -> reinterpret.(Float32, table), BITS))
    ENV["JULIA_DEBUG"] = "GameTracing"
    result = logit_solve_branch(gains)
    ENV["JULIA_DEBUG"] = ""
    @show is_error(result)
    if is_error(result)
        @show unwrap_error(result)
    end
    return nothing
end
main()
