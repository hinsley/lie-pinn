"""
experiments/wave_lie_pinn.py
--------------------------------------------------------------------
PyTorch implementation of a Lie‑PINN for the 1‑D wave equation:
    u_tt = u_xx   on  (x,t) ∈ R x [0,1]
learning an N‑dimensional continuous symmetry subgroup.

IC residuals are enforced **only when all ε parameters are zero**.

Run with:
    python experiments/wave_lie_pinn.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import plotly.graph_objects as go
import dash
# Note: If using Dash>=2.0, import directly from dash
try:
    from dash import dcc, html, Input, Output
except ImportError:
    import dash_core_components as dcc
    import dash_html_components as html
    from dash.dependencies import Input, Output

# ───────────────────────────── Hyper‑parameters ──────────────────────────────
EPOCHS          = 5_000
N_EPS           = 5
BATCH_PDE       = 512 * 4
BATCH_IC        = 256
LR              = 3e-4
EPS_RANGE       = 3.0
X_SIGMA         = 0.2  # Standard deviation for Gaussian sampling of x (infinite domain).

LAMBDA_PDE, LAMBDA_IC_DISP, LAMBDA_IC_VEL, LAMBDA_ORTHO, LAMBDA_LIE = 3.0, 10.0, 1.0, 1.0, 0.3
PRINT_EVERY = max(1, EPOCHS // 200)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

# ───────────────────────────── Network definition ────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = MLP(2 + N_EPS).to(device)
optimizer = Adam(model.parameters(), lr=LR)

# ───────────────────────────── Sampling helpers ──────────────────────────────

def sample_uniform(shape, low=0.0, high=1.0):
    return (high - low) * torch.rand(*shape, device=device) + low


def sample_xt(batch):
    # Sample x from a Gaussian centered at 0.5 for infinite-domain PINN.
    x = torch.randn((batch, 1), device=device) * X_SIGMA + 0.5
    t = sample_uniform((batch, 1))
    return x, t


def sample_eps(batch):
    return sample_uniform((batch, N_EPS), -EPS_RANGE, EPS_RANGE)

f_ic = lambda x: torch.exp(-((x - 0.5)**2) / (2 * 0.01)) # Localized Gaussian pulse centered at x=0.5.
# ───────────────────────────── Derivative helpers ────────────────────────────

def derivatives(u, x, t):
    u_x, u_t = torch.autograd.grad(u, (x, t), grad_outputs=torch.ones_like(u), create_graph=True)
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
    return u_tt, u_x, u_xx

# ───────────────────────────── Loss components ───────────────────────────────

def pde_residual():
    x, t = sample_xt(BATCH_PDE)
    eps = sample_eps(BATCH_PDE); eps.requires_grad_(True)
    x.requires_grad_(True); t.requires_grad_(True); eps.requires_grad_(True)
    u = model(torch.cat((x, t, eps), dim=1))
    u_tt, _, u_xx = derivatives(u, x, t)
    grads = [torch.autograd.grad(u, eps, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0][:, i:i+1]
             for i in range(N_EPS)]
    U = torch.cat(grads, dim=1)
    G = (U.T @ U) / BATCH_PDE
    # Extract only the strict upper triangle of G (excluding diagonal)
    mask = torch.triu(torch.ones_like(G), diagonal=1).bool()
    upper_tri = G[mask]
    return LAMBDA_PDE * (u_tt - u_xx).pow(2).mean() + LAMBDA_ORTHO * upper_tri.sum().pow(2) # - LAMBDA_LIE * torch.diag(G).sum().pow(2)


def ic_residual():
    """Initial-condition loss enforced only at ε = 0."""
    x   = sample_uniform((BATCH_IC, 1))
    t0  = torch.zeros_like(x, requires_grad=True)  # t=0 for initial slice.
    eps0 = torch.zeros((BATCH_IC, N_EPS), device=device, requires_grad=True) # Enforce only at eps=0.
    inp = torch.cat((x, t0, eps0), dim=1)
    u      = model(inp)
    res_disp = (u - f_ic(x)).pow(2).mean()
    u_t    = torch.autograd.grad(u, t0, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    res_vel  = u_t.pow(2).mean()
    return LAMBDA_IC_DISP * res_disp + LAMBDA_IC_VEL * res_vel

# ───────────────────────────── Training loop ────────────────────────────────
optimizer = Adam(model.parameters(), lr=LR)
for epoch in tqdm(range(1, EPOCHS+1), desc="Training"):  # Main training loop.
    optimizer.zero_grad()
    loss = pde_residual() + ic_residual()
    loss.backward()
    optimizer.step()
    if epoch % PRINT_EVERY == 0 or epoch == 1:
        tqdm.write(f"epoch {epoch:6d} | loss {loss.item():.6e}")

# ───────────────────── Interactive Visualization via Dash ───────────────────
print("Setting up Dash app for interactive visualization...")

# Setup common data structures for visualization (outside callback for efficiency)
xs_vis = torch.linspace(0, 1, 100, device=device)
ts_vis = torch.linspace(0, 1, 100, device=device)
X_vis, T_vis = torch.meshgrid(xs_vis, ts_vis, indexing="ij")
xs_np = xs_vis.cpu().numpy()
ts_np = ts_vis.cpu().numpy()

def compute_surface(eps_values):
    """Helper function to compute the surface grid for given epsilon values."""
    with torch.no_grad():
        eps_vec = torch.tensor(eps_values, device=device, dtype=torch.float32)
        # Ensure eps_vec has the correct shape for broadcasting if needed
        if eps_vec.dim() == 1:
            eps_vec = eps_vec.unsqueeze(0) # Add batch dim if it's just one vector
        eps_grid = eps_vec.repeat(100, 100, 1)
        inp = torch.cat((X_vis.unsqueeze(-1), T_vis.unsqueeze(-1), eps_grid), dim=-1)
        u_surface = model(inp.view(-1, 2 + N_EPS)).detach().cpu().view(100, 100).numpy()
    return u_surface

# Initialize Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    html.H1("Lie-PINN Solution u(x, t; ε)"),
    dcc.Graph(id='live-graph', style={'height': '70vh'}), # Give graph height
    html.Div([
        html.Div([
            html.Label(f"ε {i+1}", style={'paddingRight': '10px'}),
            dcc.Slider(
                id=f'eps-slider-{i}',
                min=-EPS_RANGE,
                max=EPS_RANGE,
                step=EPS_RANGE / 500, # Finer step for smoother sliding
                value=0.0,
                marks={j: f'{j:.1f}' for j in torch.linspace(-EPS_RANGE, EPS_RANGE, 11).tolist()},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'width': '80%', 'margin': 'auto', 'padding': '5px'})
        for i in range(N_EPS)
    ], style={'padding': '20px'})
])

# Define callback to update graph
@app.callback(
    Output('live-graph', 'figure'),
    [Input(f'eps-slider-{i}', 'value') for i in range(N_EPS)]
)
def update_graph(*eps_values):
    """This function is called whenever any slider value changes."""
    # eps_values will be a tuple of the current slider values
    u_surface = compute_surface(eps_values)

    # Calculate initial condition data for plotting.
    with torch.no_grad():
        ic_vals = f_ic(xs_vis).cpu().numpy()
    t0_np = torch.zeros_like(xs_vis).cpu().numpy() # t=0 for the IC line.

    # Create the figure dynamically
    fig = go.Figure(data=[
        go.Surface(
            z=u_surface.T, # Transpose Z to match Plotly's (x[j], y[i]) convention.
            x=xs_np,       # X spatial coordinate.
            y=ts_np,       # T temporal coordinate.
            colorscale="Viridis",
            cmin=-1.1, # Adjusted range slightly.
            cmax=1.1,
            showscale=False,
            name="u(x, t; ε)",
            showlegend=True
        ),
        # Add the initial condition line at t=0.
        go.Scatter3d(
            x=xs_np,  # x-values along x-axis
            y=t0_np,  # t=0 along y-axis
            z=ic_vals,
            mode='lines',
            line=dict(color='red', width=4),
            name='u(x, 0) = f_ic(x)' # Legend entry for IC.
        )
    ])

    # Apply layout settings
    fig.update_layout(
        # title="Surface changes dynamically with sliders", # Title in H1 now
        scene=dict(
            xaxis_title="x",  # Correct axis labels.
            yaxis_title="t",
            zaxis=dict(title="u", range=[-1.2, 1.2]), # Adjust range as needed
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.7), # Adjust z aspect ratio if needed.
            # Set camera view to show t=0 (y-axis) at the front.
            camera=dict(
                eye=dict(x=-1.25, y=-1.25, z=1.25) # Rotated 180 degrees around Z from default.
            )
        ),
        margin=dict(l=0, r=0, b=0, t=0), # Minimal margins
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01) # Position legend.
    )
    return fig

# Run the app
if __name__ == '__main__':
    # Make sure model is on the correct device after training
    model.to(device)
    print(f"Starting Dash server... Access at http://127.0.0.1:8050/ (Press CTRL+C to quit)")
    # Turn off reloader if running in a notebook or environment where it causes issues
    app.run(debug=True, use_reloader=False) # Use debug=False for production