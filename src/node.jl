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
    UB::Float64
    LB::Float64
end
function BeliefData(;
    observation = nothing,
    norm_const = 1.0,
    UB = Inf,
    LB = -Inf
    )
    return BeliefData(observation, norm_const, UB, LB)
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

get_UB(BN::BeliefNode) = BN.metadata.UB
get_LB(BN::BeliefNode) = BN.metadata.LB

function set_UB(BN::BeliefNode, val)
    BN.metadata.UB = val
    nothing
end

function set_LB(BN::BeliefNode, val)
    BN.metadata.LB = val
    nothing
end


function count_subchildren(parent::Node)
    # counts number of children belief nodes, including parent node
    # may be useful for random selection, and therefore needed in pruning?

    n = Int(typeof(parent) == BeliefNode)
    for child in children(parent)
        n += count_children(child)
    end
    return n
end