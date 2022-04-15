module SARSOP_julia

# when adding dependencies:
# https://pkgdocs.julialang.org/v1/creating-packages/
# skip the "generate", as the files already exist
# go into the pkg prompt, "activate .", then in the REPL "add POMDPs" for example
# this is needed to update the Project.toml dependencies.

using POMDPs
using POMDPModelTools
using POMDPPolicies

import POMDPs: Solver
import POMDPs: solve

export
    SARSOPSolver,
    solve

include("solver.jl")
include("core.jl")

end # module