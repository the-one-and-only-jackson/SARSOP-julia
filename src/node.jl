# when possible, impelement functions to get/set data so that node 
# impelementation can change, but code outside of this file stays the Sample.
# Note: set functions only matter if the value can change e.g. observation is
# a constant and only needs a get function

abstract type Node end
value(n::Node) = n.value
children(n::Node) = n.children
metadata(n::Node) = n.metadata

mutable struct BeliefData
    observation
    norm_const::Float64 # = sum(b′) = ∑_{s,s′} Z(s′,a,o) T(s,a,s′) b(s) where b(s) is the parent belief, before normalization of b′
end
function BeliefData(;
    observation = nothing,
    norm_const = 1.0
    )
    return BeliefData(observation, norm_const)
end

mutable struct ActionData
    reward::Float64
end
function ActionData(;
    reward = 0.0
    )
    return ActionData(reward)
end

struct BeliefNode <: Node
    value::DiscreteBelief
    children::Vector{Node}
    metadata::BeliefData
end

struct ActionNode{T} <: Node
    value::T
    children::Vector{BeliefNode}
    metadata::ActionData
end

# Node functions
belief(BN::BeliefNode) = BN.value.b
observation(BN::BeliefNode) = BN.metadata.observation
norm_const(BN::BeliefNode) = BN.metadata.norm_const
reward(AN::ActionNode) = AN.metadata.reward

function count_subchildren(parent::Node)
    # counts number of children belief nodes, including parent node
    # may be useful for random selection, and therefore needed in pruning?

    n = Int(typeof(parent) == BeliefNode)
    for child in children(parent)
        n += count_children(child)
    end
    return n
end