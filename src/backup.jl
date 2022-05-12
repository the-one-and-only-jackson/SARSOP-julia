function backup_all(tree, Γ, BN)
    for AN in children(BN)
        for BN′ in children(AN)
            backup_all(tree, Γ, BN′)
        end
    end
    Backup(tree, Γ, BN)
    nothing
end


# ========== Algorithm 2 ==========
# Perform α-vector backup at a node b of T_R
function Backup(tree::BeliefTree, Γ::AlphaVectorPolicy, parent::BeliefNode)
    # 1. α_{a,o} ← argmax_α (α ⋅ τ(b,a,o)), ∀ a∈𝒜, o∈𝒪
    # 2. α_a(s) ← R(s,a) + γ ∑_{o,s′} T(s,a,s′)Z(s′,a,o)α_{a,o}(s′), ∀ a∈𝒜, s∈𝒮
    # 3. α′ ← argmax(α_a ⋅ b, for a in 𝒜)
    # 4. Insert α′ into Γ.

    # https://www.overleaf.com/read/rwfcytcbvrtz
    # lightweight calcultion of optimal action, then more intensive calcultion
    # of the alpha vector correspodning to that belief/action

    pomdp, α_vectors, action_map = Γ.pomdp, Γ.alphas, Γ.action_map

    a_opt = rand(POMDPs.actiontype(pomdp))
    V = -Inf

    for AN in children(parent)
        _sum = 0.0

        for BN′ in children(AN)
            b = belief(BN′)
            _sum += norm_const(BN′) * maximum(α ⋅ b for α in α_vectors)
        end

        Q̲ = reward(AN) + discount(pomdp)*_sum

        if Q̲ > V
            V = Q̲
            a_opt = value(AN)
        end
    end

    set_LB(parent, V)

    push!(action_map, a_opt)
    push!(α_vectors, calc_α(Γ, parent, a_opt))
    

    return AlphaVectorPolicy(pomdp, α_vectors, action_map)
end


function calc_α(Γ::AlphaVectorPolicy, parent::BeliefNode, a)
    belief = value(parent)
    pomdp, 𝒮, b = belief.pomdp, belief.state_list, belief.b

    α′ = similar(b)

    AN = get_ActionNode!(parent, a)

    for (si,s) in enumerate(𝒮) # this may need to be refined in the future
        _sum = 0.0
        for (s′,T) in weighted_iterator(transition(pomdp,s,a))
            Z = POMDPs.observation(pomdp,a,s′)
            for BN in children(AN)
                o = observation(BN)
                b′ = BN.value.b
                α_ao = argmax(α->α⋅b′, Γ.alphas)
                _sum += T * pdf(Z,o) * α_ao[POMDPs.stateindex(pomdp,s′)]
            end
        end

        α′[si] = POMDPs.reward(pomdp,s,a) + POMDPs.discount(pomdp) * _sum
    end

    return α′
end