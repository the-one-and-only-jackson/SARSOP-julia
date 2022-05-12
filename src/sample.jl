function Sample(solver::SARSOPSolver, tree::BeliefTree, Γ::AlphaVectorPolicy)
    ϵ = solver.ϵ
    
    L = maximum(α ⋅ belief(tree.root) for α in Γ.alphas)
    U = L + ϵ 
    
    SamplePoints(tree, Γ, tree.root, L, U, ϵ, 1)
end

function SamplePoints(tree::BeliefTree, Γ::AlphaVectorPolicy, BN::BeliefNode, L, U, ϵ, t)
    (V̄, a′) = get_UB(tree, BN)
    V̲ = maximum(α->α⋅belief(BN), Γ.alphas)

    pomdp = Γ.pomdp
    γ = discount(pomdp)

    if V̄ ≤ max(U, V̲ + ϵ*γ^(-t))
        return
    end

    L′ = max(L, V̲)
    U′ = max(U, V̲ + γ^(-t)*ϵ)

    b = belief(BN)
    b′ = similar(b)
    o′ = observation(BN)
    K = 0.0 # p(o′|b,a′)
    max_val = -Inf
    Γ_upper = tree.qmdp_policy
    L_sum = 0.0
    U_sum = 0.0
    V̄ = 0.0
    V̲ = 0.0

    for o in POMDPs.observations(pomdp)
        (b′_temp, K_temp) = τ(BN, a′, o)

        temp_V̲ = maximum(α->α⋅b′_temp, Γ.alphas)
        temp_V̄ = maximum(α->α⋅b′_temp, Γ_upper.alphas)

        temp_val = K_temp*(temp_V̄-temp_V̲) 

        L_sum += K_temp * temp_V̲
        U_sum += K_temp * temp_V̄

        if temp_val > max_val
            b′ = b′_temp
            o′ = o
            K  = K_temp
            V̲  = temp_V̲
            V̄  = temp_V̄
            max_val = temp_val
        end
    end

    L_sum -= K * V̲
    U_sum -= K * V̄

    𝒮 = value(BN).state_list
    R = sum(b_s*POMDPs.reward(pomdp,s,a′) for (s,b_s) in zip(𝒮, b))

    L_t = ((L′-R)/γ - L_sum)/K
    U_t = ((U′-R)/γ - U_sum)/K

    BN′ = insert_BeliefNode!(tree, BN, b′, a′, o′, K)

    SamplePoints(tree, Γ, BN′, L_t, U_t, ϵ, t+1)

    return
end


get_UB(tree::BeliefTree, BN::BeliefNode) = get_UB(tree, belief(BN))
function get_UB(tree::BeliefTree, b)
    Γ = tree.qmdp_policy.alphas
    𝒜 = tree.qmdp_policy.action_map

    (V, a_idx) = findmax(α ⋅ b for α in Γ)
    a = 𝒜[a_idx]

    return (V,a)
end


function τ(BN::BeliefNode, a, o)
    belief = value(BN)
    pomdp, 𝒮, b = belief.pomdp, belief.state_list, belief.b

    b′ = similar(b)
    K = 0.0
    for (si′,s′) in enumerate(𝒮)
        _sum = 0.0
        for (s,b_s) in zip(𝒮,b)
            _sum += pdf(transition(pomdp,s,a),s′) * b_s
        end
        b′[si′] = pdf(POMDPs.observation(pomdp,a,s′),o) * _sum
        K += b′[si′]
    end
    b′ /= K
    
    return (b′, K)
end