using Plots
function alpha_plot(Γ)
    p = plot(plot(xlims=[0,1], ylims=[-80,30]))
    for (a,α) ∈ zip(Γ.action_map, Γ.alphas)
        plot!(p, [0,1], α, label=a)
    end
    display(p)
end


# ========== Algorithm 1 ==========
function solve(solver::SARSOPSolver, pomdp::POMDP)
    # 1. Initialize the set Γ of α-vectors, representing the lower bound V̲ on the 
    #       optimal value function V∗. Initialize the upper bound V̄ on V∗.
    # 2. Insert the initial belief point b0 as the root of the tree T_R.
    # 3. repeat 
    # 4. SAMPLE(T_R, Γ)
    # 5.    Choose a subset of nodes from T_R. For each chosen node b, BACKUP(T_R,Γ,b).
    # 6.    PRUNE(T_R, Γ)
    # 7. until termination conditions are satisfied
    # 8. return Γ

    Γ = alpha_init(pomdp)
    tree = BeliefTree(pomdp)

    start_time = time_ns()
    count = 0
    while time_ns()-start_time < solver.max_time
        count += 1
        println(count)

        Sample(solver, tree, Γ)
        backup_all(tree, Γ, tree.root)
        Γ = PRUNE(tree, Γ)

        alpha_plot(Γ)
        println("nodes: $(tree.n_nodes)")
    end

    return Γ # return alpha vectors and corresponding actions
end

function alpha_init(pomdp)
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    γ = discount(pomdp)
    r = StateActionReward(pomdp)
    
    α_init = 1 / (1 - γ) * maximum(minimum(r(s, a) for s in S) for a in A)
    Γ = [fill(α_init, length(S)) for a in A]

    return AlphaVectorPolicy(pomdp, Γ, A)
end

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


# ========== Algorithm 3 ==========
# Sampling near R*
function Sample(solver::SARSOPSolver, tree::BeliefTree, Γ::AlphaVectorPolicy)
    # 1. Set L to the current lower bound on the value function at the root b_0 of T_R. 
    #    Set U to L + ϵ, where ϵ is the current target gap size at b0.
    # 2. SAMPLEPOINTS(T_R, Γ, b_0, L, U, ϵ, 1).

    ϵ = solver.ϵ
    L = maximum(α ⋅ belief(tree.root) for α in Γ.alphas)
    U = L + ϵ 
    
    SamplePoints(tree, Γ, tree.root, L, U, ϵ, 1)
end

function SamplePoints(tree::BeliefTree, Γ::AlphaVectorPolicy, BN::BeliefNode, L, U, ϵ, t)
    # 3. Let V̂ be the predicted value of V*(b).
    # 4. if V̂ ≤ L and V̄ ≤ max{U, V̲(b) + ϵγ^{-t}} then
    # 5.    return
    # 6. else
    # 7.    Q̲ ← max_a Q̲(b,a)
    # 8.    L′ ← max{L, Q̲}
    # 9.    U′ ← max{U, Q̲ + γ^{-t} ϵ}
    # 10.   a′ ← argmax_a Q̄(b,a)
    # 11.   o′ ← argmax_o p(o|b,a′) (V̄(τ(b,a′,o)) - V̲(τ(b,a′,o)))
    # 12.   Calculate L_t so that 
    #           L′ = ∑_s R(s,a′)b(s) + γ ( p(o′|b,a′)L_t + ∑_{o≠o′} p(o|b,a′)V̲(τ(b,a′,o)) )
    # 13.   Calculate U_t so that 
    #           U′ = ∑_s R(s,a′)b(s) + γ ( p(o′|b,a′)U_t + ∑_{o≠o′} p(o|b,a′)V̲(τ(b,a′,o)) )
    # 14.   b′ ← τ(b,a′,o′)
    # 15.   Insert b′ into T_R as a child of b.
    # 16.   SamplePoints(T_r, Γ, b′, L_t, U_t, ϵ, t+1)

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

# ========== needed functions ==========

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


# ========== helper functions ==========

function Q(Γ::AlphaVectorPolicy,b,a)
    α_vectors, action_map = Γ.alphas, Γ.action_map
    if a ∉ action_map
        return -Inf
    end
    idx = action_map .== a
    return maximum(α->α⋅b, α_vectors[idx])
end
Q(Γ::AlphaVectorPolicy,b::BeliefNode,a) = Q(Γ,belief(b),a)

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