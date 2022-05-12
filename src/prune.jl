function PRUNE(tree::BeliefTree, Γ::AlphaVectorPolicy) # not sure what parameters needed
    Γ = prune_alphas(Γ)
    prune_actions!(tree, tree.root, Γ)
    return Γ
end

function prune_alphas(Γ)
    new_actions = Vector{eltype(Γ.action_map)}(undef,0)
    new_alphas = Vector{eltype(Γ.alphas)}(undef,0)

    for (a,α) in zip(Γ.action_map, Γ.alphas)
        if α ∈ new_alphas
            continue
        end

        push!(new_actions, a)
        push!(new_alphas, α)
        for β in Γ.alphas
            if α[1]<β[1] && α[2]<β[2]
                pop!(new_actions)
                pop!(new_alphas)
                break
            end
        end
    end

    return AlphaVectorPolicy(Γ.pomdp, new_alphas, new_actions)
end

function prune_actions!(tree, BN, Γ)
    b = belief(BN)
    𝒜 = unique(tree.qmdp_policy.action_map)

    Q̄ = [Q(tree.qmdp_policy, b, a) for a in 𝒜]

    for AN in children(BN)
        a = value(AN)
        Q̲ = Q(Γ, b, a)
        for a in 𝒜[Q̲ .> Q̄]
            prune_tree(tree, BN, a)
        end
    end

    for AN in children(BN)
        for BN′ in children(AN)
            prune_actions!(tree, BN′, Γ)
        end
    end

    nothing
end


function Q(Γ::AlphaVectorPolicy,b,a)
    α_vectors, action_map = Γ.alphas, Γ.action_map
    if a ∉ action_map
        return -Inf
    end
    idx = action_map .== a
    return maximum(α->α⋅b, α_vectors[idx])
end
Q(Γ::AlphaVectorPolicy,b::BeliefNode,a) = Q(Γ,belief(b),a)

