abstract type Node end
value(n::Node) = n.value
children(n::Node) = n.children
metadata(n::Node) = n.metadata

mutable struct BeliefData
    observation
    norm_const::Float64 # = p(o|b,a)
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
    n = Int(typeof(parent) == BeliefNode)
    for child in children(parent)
        n += count_children(child)
    end
    return n
end