# experiments/heat-pinn.jl
# Solve the 1D heat equation ut = uxx on x in [0,1], t in [0,1] using a PINN
# with NeuralPDE.jl.
# Plot the resulting solution with GLMakie.

using Pkg
Pkg.activate(".")
Pkg.instantiate()

# Import necessary packages.
using DiffEqFlux
using Flux
using GLMakie
using ModelingToolkit
using NeuralPDE
using ProgressMeter
using Random
using Lux

# Define the spatial and temporal variables for the PDE.
@parameters x t
@variables u(..)

# Define derivative operators.
Dx = Differential(x)
Dt = Differential(t)

# Define the PDE: ∂u/∂t = ∂²u/∂x².
pde = Dt(u(x, t)) ~ Dx(Dx(u(x, t)))

# Define the initial condition: u(x, 0) = sin(pi*x).
f(x) = sin(pi * x)
ic = u(x, 0) ~ f(x)

# Define boundary conditions: u(0, t) = 0, u(1, t) = 0.
g_left(t) = 0.0
g_right(t) = 0.0
bc_left = u(0, t) ~ g_left(t)
bc_right = u(1, t) ~ g_right(t)

# Define the domain for x and t.
domains = [x ∈ IntervalDomain(0.0, 1.0), t ∈ IntervalDomain(0.0, 1.0)]

# Define the PDE system with the equation, conditions, domains, variables, and function.
pde_system = PDESystem(
  [pde],                                # wrap in a Vector
  [ic, bc_left, bc_right],
  domains,
  [x, t],
  [u];
  name = :heat_1d                      # required keyword
)

# Define a neural network: 2 inputs (x, t), 50 hidden neurons with tanh, 1 output (u).
chain = Lux.Chain(Lux.Dense(2, 50, tanh), Lux.Dense(50, 1))
# Get initial parameters and state (needed for Lux evaluation).
rng = Random.default_rng()
ps, st = Lux.setup(rng, chain)

# Choose a PINN discretization strategy: random sampling via quadrature.
strategy = PhysicsInformedNN(chain, QuadratureTraining())

# Discretize the PDE system into an optimization problem for the neural network.
discretized_pde = discretize(pde_system, strategy)

# Train the PINN using DiffEqFlux's ADAM optimizer (learning rate 0.01) for 3000 iterations with a progress meter.
maxiters = 1000  # Total number of optimization steps.
pbar = Progress(maxiters, desc = "Training PINN")  # create a progress bar.
cb = (itr, loss) -> begin
    next!(pbar; showvalues = [(:loss, loss)])
    return false
end
res = NeuralPDE.solve(discretized_pde, ADAM(0.01); maxiters = maxiters, callback = cb)

# The neural network `chain` structure is defined above.
# Define a helper function to evaluate the solution at any point (x, t) using the trained parameters from `res.u`.
# Note: We use the original `chain`, the optimal parameters `res.u`, and the initial state `st`.
u_pred(x_val, t_val) = chain([x_val, t_val], res.u, st)[1][1]

# Prepare a grid of points in the domain to visualize the solution.
xs = range(0.0, 1.0, length = 100)
ts = range(0.0, 1.0, length = 100)
u_values = [u_pred(x, t) for x in xs, t in ts]

# Create a 3D surface plot of u(x, t) using GLMakie.
begin
  fig = Figure(size = (800, 600))
  ax = Axis3(fig[1, 1], title = "PINN Solution of 1D Heat Equation", xlabel = "x", ylabel = "t", zlabel = "u(x, t)")
  surface!(ax, xs, ts, u_values, colormap = :thermal, transparency = true, alpha = 0.9)
  # Display the interactive figure.
  display(fig)
end