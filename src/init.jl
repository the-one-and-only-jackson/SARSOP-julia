function alpha_init(pomdp)
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    γ = discount(pomdp)
    r = StateActionReward(pomdp)
    
    α_init = 1 / (1 - γ) * maximum(minimum(r(s, a) for s in S) for a in A)
    Γ = [fill(α_init, length(S)) for a in A]

    return AlphaVectorPolicy(pomdp, Γ, A)
end

function BeliefTree(pomdp::POMDP)
    belief = uniform_belief(pomdp)
    children = Vector{ActionNode{actiontype(pomdp)}}()
    metadata = BeliefData(observation=rand(POMDPs.observations(pomdp))) # metadata update here
    BN = BeliefNode(belief, children, metadata)

    # qmdp upper bound
    solver = QMDPSolver()
    policy = solve(solver, pomdp)

    return BeliefTree(BN, 1, policy)
end