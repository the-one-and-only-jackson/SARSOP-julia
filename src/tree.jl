# search for "# metadata update here"

mutable struct BeliefTree
    root::Node
    n_nodes::Int
    qmdp_policy::AlphaVectorPolicy
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

function get_ActionNode!(parent::BeliefNode, a)
    for AN in children(parent) # check if action exists
        if value(AN) == a
            return AN
        end
    end

    # if action node does not exist, create it
    belief = value(parent)
    pomdp, 𝒮, b = belief.pomdp, belief.state_list, belief.b
    R = sum(b*POMDPs.reward(pomdp,s,a) for (s,b) in zip(𝒮, b))

    metadata = ActionData(reward = R) # metadata update here
    AN = ActionNode(a, Vector{BeliefNode}(), metadata)
    push!(children(parent), AN)

    return AN
end

function insert_BeliefNode!(tree::BeliefTree, parent::BeliefNode, b′, a, o, K) # creates a child belief node
    AN = get_ActionNode!(parent, a)
    
    for BN in AN.children # check if belief already exists
        if observation(BN) == o
            return BN
        end
    end

    pomdp, 𝒮 = value(parent).pomdp, value(parent).state_list

    belief = DiscreteBelief(pomdp,𝒮,b′)
    children = Vector{ActionNode{actiontype(pomdp)}}()
    UB = maximum(α ⋅ b′ for α in tree.qmdp_policy.alphas)
    metadata = BeliefData(observation=o, norm_const=K, UB=UB) # metadata update here

    BN = BeliefNode(belief, children, metadata)
    push!(AN.children, BN)
    tree.n_nodes += 1
    
    # populate BN's children (to see reward)
    for a in actions(pomdp, b′)
        get_ActionNode!(BN, a)
    end

    return BN
end

function prune_tree(tree::BeliefTree, parent::BeliefNode, a)
    # Prunes an action branch of the belief tree
    # a is an action, not an action node

    # I think recusion will help the garbage collector, implement this later

    for AN in children(parent)
        if value(AN) == a
            tree.n_nodes -= count_subchildren(AN)

            new_children = []
            for AN′ in children(parent)
                if AN != AN′
                    push!(new_children, AN′)
                end
            end
            parent.children = new_children

            break
        end
    end

    nothing
end