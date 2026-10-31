# The signatures Corset's `inferred_methods` types, found by walking from the package's entry points
# along `:invoke` edges in optimized IR. They depend on the entry points alone, not on what ran before.

"""Whether `signature` names a type defined in `target` anywhere in its parameters."""
function mentions_module(signature::Any, target::Module, depth::Integer)::Bool
    if depth > 6
        return false
    elseif signature isa UnionAll
        return mentions_module(signature.body, target, depth + 1)
    elseif signature isa Union
        return mentions_module(signature.a, target, depth + 1) ||
            mentions_module(signature.b, target, depth + 1)
    elseif !(signature isa DataType)
        return false
    end
    return parentmodule(signature) === target ||
        any(
            parameter -> mentions_module(parameter, target, depth + 1),
            signature.parameters,
        )
end

"""The method instance an `:invoke` statement names (a `CodeInstance` on Julia 1.12 and later), or `nothing`."""
invoked_instance(callee::Core.MethodInstance)::Core.MethodInstance = callee
invoked_instance(callee::Core.CodeInstance)::Union{
    Nothing,
    Core.MethodInstance,
} = invoked_instance(callee.def)
invoked_instance(::Any)::Nothing = nothing

"""The signatures of `target`'s frames (and frames carrying its types) that `signature` invokes."""
function invoked_signatures(signature::Any, target::Module)::Vector{Any}
    typed = Base.code_typed_by_type(signature; optimize = true)
    if isempty(typed)
        return Any[]
    end
    code = first(first(typed)).code
    invoked = [
        invoked_instance(first(statement.args))
        for statement in code
        if statement isa Expr && statement.head in (:invoke, :invoke_modify)
    ]
    return Any[
        callee.specTypes
        for callee in invoked
        if callee isa Core.MethodInstance &&
            callee.def isa Method &&
            (callee.def.module === target || mentions_module(callee.specTypes, target, 0))
    ]
end

"""Every signature reachable from `seeds` along the invoke edges that stay in `target`."""
function walked_signatures(seeds::AbstractVector, target::Module)::Vector{Type}
    seen = Set{Any}()
    frontier = unique(seeds)
    while !isempty(frontier)
        union!(seen, frontier)
        frontier = unique(
            callee
            for signature in frontier
            for callee in invoked_signatures(signature, target)
            if !(callee in seen)
        )
    end
    return Type[signature for signature in seen]
end

"""The signature of `callee(args...; options...)`, through `Core.kwcall` when keywords are given."""
function entry_signature(::F, args::Vararg{Any, N}; options...)::Type{<:Tuple} where {F, N}
    positional = Any[F, map(typeof, args)...]
    if isempty(options)
        return Tuple{positional...}
    end
    return Tuple{typeof(Core.kwcall), typeof(values(options)), positional...}
end
