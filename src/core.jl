# ========== Algorithm 1 ==========
function SARSOP_main(solver::SARSOPSolver, pomdp::POMDP)
    # 1. Initialize the set Γ of α-vectors, representing the lower bound V on the optimal value function V∗. Initialize the upper bound V̄ on V∗.
    # 2. Insert the initial belief point b0 as the root of the tree T_R.
    # 3. repeat 
    # 4. SAMPLE(T_R, Γ)
    # 5.    Choose a subset of nodes from T_R. For each chosen node b, BACKUP(T_R,Γ,b).
    # 6.    PRUNE(T_R, Γ)
    # 7. until termination conditions are satisfied
    # 8. return Γ


    # how to initialize lower bound alpha vectors?
    Γ = AlphaVectorPolicy(pomdp, α_vectors, action_map)

    # how to initialize upper bound V̄?
    # V̄ = 

    # how to represent belief tree?
    # T_r = 

    terminal_condition = false
    while !terminal_condition
        Sample(T_R, Γ)

        subset_T_R = [] # build this somehow
        for b in subset_T_R
            Backup(T_R, Γ, b)
        end

        PRUNE(T_R, Γ)
    end

    return Γ # return alpha vectors and corresponding actions
end

# ========== Algorithm 2 ==========
# Perform α-vector backup at a node b of T_R
function Backup(T_R, Γ, b)
    # 1. α_{a,o} ← argmax_α (α ⋅ τ(b,a,o)), ∀ a∈𝒜, o∈𝒪
    # 2. α_a(s) ← R(s,a) + γ ∑_{o,s′} T(s,a,s′)Z(s′,a,o)α_{a,o}(s′), ∀ a∈𝒜, s∈𝒮
    # 3. α′ ← argmax(α_a ⋅ b, for a in 𝒜)
    # 4. Insert α′ into Γ.

    # CURRENT ISSUE:
    # This should only be looking at the children of B, rather than the entire pomdp.
    # This is not stated in the algorithm, but rather in Section III.B 
    # Will need to reasses how to compute. Rather than iterate over all actions, observations,
    # should instead iterate over children of b.

    pomdp, α_vectors, action_map = Γ.pomdp, Γ.α_vectors, Γ.action_map

    𝒜 = ordered_actions(pomdp)
    𝒮 = ordered_states(pomdp)
    γ = discount(pomdp)

    α = zeros( length(𝒮), length(𝒜) )
    for (ai, a) in 𝒜
        for (si, s) in 𝒮
            temp_sum = 0.0
            for (s′, T) in weighted_iterator(transition(pomdp, s, a))
                for (o, Z) in weighted_iterator(observation(pomdp, a, s′))
                    b′ = τ(b,a,o)
                    temp_sum += T * Z * argmax_(α_ -> α_ ⋅ b′, α_vectors)
                end
            end
            α[si, ai] = reward(pomdp,s,a) + γ * temp_sum
        end
    end

    idx = argmax( vec( b' * α ) ) # vec vs transpose for speed?
    push!(α_vectors, α[:, idx])
    push!(action_map, 𝒜[idx])

    return AlphaVectorPolicy(pomdp, α_vectors, action_map)
end

# ========== Algorithm 3 ==========
# Sampling near R*
function Sample(T_R, Γ)
    # 1. Set L to the current lower bound on the value function at the root b_0 of T_R. Set U to L + ϵ, where ϵ is the current target gap size at b0.
    # 2. SAMPLEPOINTS(T_R, Γ, b_0, L, U, ϵ, 1).
end

function SamplePoints(T_r, Γ, b, L, U, ϵ, t)
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

function PRUNE(T_R, Γ)
    # see section III.D
end

# belief update - section III.B
# implement using DiscreteUpater package, as in HW 6?
function τ(b,a,o)
    # b′(s′) = τ(b,a,o) = η Z(s′,a,o) ∑_s T(s,a,s′)b(s)
    return b′
end



# ========== helper function ==========
argmax_(f, domain) = domain[argmax(f, domain)]