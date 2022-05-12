using Revise
using SARSOP_julia
using POMDPModels: TigerPOMDP
using Plots
using SARSOP

function alpha_plot(Γ)
    p = plot(xlims=[0,1], ylims=[-80,30])
    for (a,α) ∈ zip(Γ.action_map, Γ.alphas)
        plot!(p, [0,1], α, label=a)
    end
    p
end

true_policy = solve(SARSOP.SARSOPSolver(), TigerPOMDP())
alpha_plot(true_policy)

solver = SARSOP_julia.SARSOPSolver(ϵ=0.001, max_time=0_100_000_000) # 10 seconds
results = solve(solver, TigerPOMDP())
alpha_plot(results)