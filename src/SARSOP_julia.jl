module SARSOP_julia

# when adding dependencies:
# https://pkgdocs.julialang.org/v1/creating-packages/
# skip the "generate", as the files already exist
# go into the pkg prompt, "activate .", then in the REPL "add POMDPs" for example
# this is needed to update the Project.toml dependencies.

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
include("core.jl")

end

###########


using POMDPModels: TigerPOMDP

solver = SARSOPSolver(ϵ=1.0, max_time=10_000_000_000) # 10 seconds
results = solve(solver, TigerPOMDP())