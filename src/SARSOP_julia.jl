module SARSOP_julia

using POMDPs
using POMDPModelTools
using POMDPPolicies
using BeliefUpdaters
using LinearAlgebra
using QMDP

import POMDPs: Solver
import POMDPs: solve

export
    SARSOPSolver,
    solve

struct SARSOPSolver <: Solver
    ϵ::Float64
    max_time::UInt64 # time in ns
end
SARSOPSolver(; ϵ=1e-3, max_time=60) = SARSOPSolver(ϵ, max_time)

include("node.jl")
include("tree.jl")
include("init.jl")
include("sample.jl")
include("backup.jl")
include("prune.jl")

function solve(solver::SARSOPSolver, pomdp::POMDP)
    Γ = alpha_init(pomdp)
    tree = BeliefTree(pomdp)

    start_time = time_ns()
    while time_ns()-start_time < solver.max_time
        Sample(solver, tree, Γ)
        backup_all(tree, Γ, tree.root)
        Γ = PRUNE(tree, Γ)
    end

    return Γ # return alpha vectors and corresponding actions
end

end