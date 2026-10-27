# How Julia parses a short-form method whose return annotation is followed by `where`.
function main()::Nothing
    println("VERSION ", VERSION)
    for source in (
        "f(x::A{E})::Vector{E} where {E} = 1",
        "f(x::T)::T where {T} = x",
        "(f(x::T)::T) where {T} = x",
        "function f(x::T)::T where {T}\n x\nend",
    )
        println("== ", repr(source))
        parsed = Meta.parse(source)
        Meta.show_sexpr(stdout, parsed)
        println()
        dump(parsed; maxdepth = 6)
        defined = try
            Core.eval(Module(), parsed)
            "defines"
        catch error
            sprint(showerror, error)
        end
        println("eval: ", defined)
    end
    return nothing
end

main()
