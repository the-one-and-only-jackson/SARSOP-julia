# ========== Algorithm 1 ==========
function SARSOP_main(solver::SARSOPSolver, pomdp::POMDP)
    # 1. Initialize the set Γ of α-vectors, representing the lower bound V̲ on the optimal value function V∗. Initialize the upper bound V̄ on V∗.
    # 2. Insert the initial belief point b0 as the root of the tree T_R.
    # 3. repeat 
    # 4. SAMPLE(T_R, Γ)
    # 5.    Choose a subset of nodes from T_R. For each chosen node b, BACKUP(T_R,Γ,b).
    # 6.    PRUNE(T_R, Γ)
    # 7. until termination conditions are satisfied
    # 8. return Γ


    # how to initialize lower bound alpha vectors?
    # α_vectors = ?????
    # action_map = ?????
    Γ = AlphaVectorPolicy(pomdp, α_vectors, action_map)

    # how to initialize upper bound V̄?
    # possibly with QMDP? the sarsop paper seems to mention this, 
    # but im not sure how this is a guaranteed upper bound
    # another issue: where do we store this? in the tree perhaps?
    # each node will need an upper bound, but that must be assigned later
    # V̄ = 

    
    b0 = ones(length(states(pomdp)))/length(states(pomdp)) # uniform weighted, probably wrong
    tree = Tree(b0)

    terminal_condition = false
    while !terminal_condition
        Sample(tree, Γ)

        subset_tree = [] # build this somehow
        for b in subset_tree
            Backup(tree, Γ, b)
        end

        # possibly: 
        # for ii in rand(1:tree.n_nodes)
        #     b = selectNode(tree, ii) # this function would need to be made, not difficult
        #     Backup(tree, Γ, b)
        # end

        PRUNE(tree, Γ)
    end

    return Γ # return alpha vectors and corresponding actions
end

# ========== Algorithm 2 ==========
# Perform α-vector backup at a node b of T_R
function Backup(T_R::BeliefTree, Γ::AlphaVectorPolicy, parent::BeliefNode)
    # 1. α_{a,o} ← argmax_α (α ⋅ τ(b,a,o)), ∀ a∈𝒜, o∈𝒪
    # 2. α_a(s) ← R(s,a) + γ ∑_{o,s′} T(s,a,s′)Z(s′,a,o)α_{a,o}(s′), ∀ a∈𝒜, s∈𝒮
    # 3. α′ ← argmax(α_a ⋅ b, for a in 𝒜)
    # 4. Insert α′ into Γ.

    # https://www.overleaf.com/read/rwfcytcbvrtz
    # lightweight calcultion of optimal action, then more intensive calcultion
    # of the alpha vector correspodning to that belief/action

    pomdp, α_vectors, action_map = Γ.pomdp, Γ.α_vectors, Γ.action_map

    a_opt = rand(POMDPs.actiontype(pomdp))
    V = -Inf

    for AN in children(parent)
        _sum = 0.0

        for BN′ in children(AN)
            b = belief(BN′)
            _sum += norm_const(BN′) * maximum(α ⋅ b for α in Γ)
        end

        _sum = reward(AN) + discount(pomdp)*_sum

        if _sum > V
            V = _sum
            a_opt = value(AN)
        end
    end

    push!(action_map, a_opt)
    push!(α_vectors, calc_α(Γ, parent, a_opt))
    

    return AlphaVectorPolicy(pomdp, α_vectors, action_map)
end

function calc_α(Γ::AlphaVectorPolicy, parent::BeliefNode, a)
    belief = value(parent)
    pomdp, 𝒮, b = belief.pomdp, belief.state_list, belief.b

    α′ = similar(b)

    AN = insert_ActionNode!(parent, a)

    for (si,s) in enumerate(𝒮) # this may need to be refined in the future
        _sum = 0.0
        for (s′,T) in weighted_iterator(transition(pomdp,s,a))
            Z = POMDPs.observation(pomdp,a,s′)
            for BN in children(AN)
                o = observation(BN)
                b′ = belief(BN)
                _sum += T * pdf(Z,o) * argmax_(α->α⋅b′, Γ)
            end
        end

        α′[si] = POMDPs.reward(pomdp,s,a) + POMDPs.discount(pomdp) * _sum
    end

    return α′
end

# ========== Algorithm 3 ==========
# Sampling near R*
function Sample(T_R::BeliefTree, Γ::AlphaVectorPolicy)
    # 1. Set L to the current lower bound on the value function at the root b_0 of T_R. Set U to L + ϵ, where ϵ is the current target gap size at b0.
    # 2. SAMPLEPOINTS(T_R, Γ, b_0, L, U, ϵ, 1).

    b = T_R.b0.b
    ϵ = 0 # ??????????????????? wtf is this value supposed to be, definitely not zero
    L = maximum(α ⋅ T_R.b0 for α in Γ)
    U = L + ϵ 
    
    SAMPLEPOINTS(T_R, Γ, b, L, U, ϵ, 1)
end

function SamplePoints(T_R::BeliefTree, Γ::AlphaVectorPolicy, b, L, U, ϵ, t)
    # 3. Let V̂ be the predicted value of V*(b).
    # 4. if V̂ ≤ L and V̄ ≤ max{U, V̲(b) + ϵγ^{-t}} then
    # 5.    return
    # 6. else
    # 7.    Q̲ ← max_a Q̲(b,a)
    # 8.    L′ ← max{L, Q̲}
    # 9.    U′ ← max{U, Q̲ + γ^{-t} ϵ}
    # 10.   a′ ← argmax_a Q̄(b,a)
    # 11.   o′ ← argmax_o p(o|b,a′) (V̄(τ(b,a′,o)) - V̲(τ(b,a′,o)))
    # 12.   Calculate L_t so that L′ = ∑_s R(s,a′)b(s) + γ ( p(o′|b,a′)L_t + ∑_{o≠o′} p(o|b,a′)V̲(τ(b,a′,o)) )
    # 13.   Calculate U_t so that U′ = ∑_s R(s,a′)b(s) + γ ( p(o′|b,a′)U_t + ∑_{o≠o′} p(o|b,a′)V̲(τ(b,a′,o)) )
    # 14.   b′ ← τ(b,a′,o′)
    # 15.   Insert b′ into T_R as a child of b.
    # 16.   SamplePoints(T_r, Γ, b′, L_t, U_t, ϵ, t+1)
end

# ========== needed functions ==========

function PRUNE(tree::BeliefTree, Γ::AlphaVectorPolicy) # not sure what parameters needed h;ere
    # see section III.D

    # insert code to determine which belief-action pair to prune?

    # BN = BeliefNode
    # a = action (not ActionNode)
    prune_tree(tree, BN, a)

    nothing
end



# ========== helper functions ==========
argmax_(f, domain) = domain[argmax(f, domain)]