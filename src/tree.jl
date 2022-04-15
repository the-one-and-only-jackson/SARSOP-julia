# Based on Fig. 2

# note: "b" could either mean the node, or the actual belief vector depending on context

# NONE OF THIS IS TESTED, I JUST WANTED TO GET SOMETHING DONE TODAY.
# TODO: 
#   1. Make sure this is all that is required of the tree
#   2. Add additional helper functions if needed
#   3. Add tests to make sure this works

const bType = Vector # this is the structure of the belief (NOT the belief node), can be changed if wanted

mutable struct Tree
    b0::BeliefNode
    n_nodes::Int # number of belief nodes
end
Tree(b0) = Tree(BeliefNode(b0), 1)

# possibly add type constraints later? particularly for the dicts
struct BeliefNode
    b::bType # belief vector, make sure to use ordered states when constructing
    actionNodes::Dict # actionNodes[action] = ActionNode
end
BeliefNode(b) = BeliefNode(b, Dict())

struct ActionNode
    a
    beliefNodes::Dict # beliefNodes[observation] = BeliefNode
end
ActionNode(a) = ActionNode(a, Dict())


function insert!(tree::Tree, parent::BeliefNode, b::bType, a, o) # creates a child belief node
    AN = get!(parent.actionNodes, a, ActionNode(a)) # is it faster to write if-else here?
    
    if !haskey(AN, o) # if (b′,a′,o′) exists, do nothing (is this correct?) 
        AN.beliefNodes[o] = BeliefNode(b)
        tree.n_nodes += 1
    end

    nothing
end

function count_children(parent::BeliefNode)
    # counts number of children belief nodes
    # may be useful for random selection, and therefore pruning

    n = 0
    for AN in values(parent)
        for BN in values(AN.beliefNodes)
            n += count_children(BN)
        end
    end
    return n
end

function prune_tree(tree::Tree, BN::BeliefNode, a)
    # Recursively prunes branch corresponding to taking action a at belief node BN.
    # I think the recusion will help the garbage collector, if this is not true,
    # can just count_children and delete the action node.

    if haskey(BN, a)
        AN = BN.actionNodes[a]

        for (o,BN′) in pairs(AN.beliefNodes)
            for a′ in keys(BN′.actionNodes)
                prune_tree(tree, BN′, a′)
            end

            delete!(AN, o)
            tree.n_nodes -= 1
        end

        delete!(BN, a)
    end

    nothing
end