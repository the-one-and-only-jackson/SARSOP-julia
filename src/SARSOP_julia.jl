module SARSOP_julia

using POMDPs
using POMDPModelTools
using POMDPPolicies

import POMDPs: Solver
import POMDPs: solve

export
    SARSOPSolver,
    solve

include("core.jl")

# Add sovler parameters as needed, making sure to specify types when possible.
# When adding a parameter, update the helper function
struct SARSOPSolver <: Solver
    param_1
end
function SARSOPSolver(; param_1=default_val_1)
    return SARSOPSolver(param_1)
end

function solve(solver::SARSOPSolver, pomdp::POMDP)
    return SARSOP_main(solver, pomdp)
end

end # module