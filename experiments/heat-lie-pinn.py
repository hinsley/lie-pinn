"""
experiments/heat_lie_pinn.py
--------------------------------------------------------------------
PyTorch implementation of a Lie‑PINN for the 1‑D heat equation:
    u_t = u_xx   on  (x,t) ∈ [0,1]²
learning an N‑dimensional continuous symmetry subgroup.

IC/BC residuals are enforced **only when all ε parameters are zero** as per
user request.

Run with:
    python experiments/heat_lie_pinn.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import plotly.graph_objects as go
from itertools import product
import dash
# Note: If using Dash>=2.0, import directly from dash
try:
    from dash import dcc, html, Input, Output
except ImportError:
    import dash_core_components as dcc
    import dash_html_components as html
    from dash.dependencies import Input, Output

# ───────────────────────────── Hyper‑parameters ──────────────────────────────
EPOCHS        = 10_000
N_EPS         = 2
BATCH_PDE     = 256
BATCH_IC      = 128
BATCH_BC      = 128
BATCH_ORTHO   = 256
LR            = 3e-4
EPS_RANGE     = 1.0

LAMBDA_PDE, LAMBDA_IC, LAMBDA_BC, LAMBDA_ORTHO = 1.0, 10.0, 10.0, 1.0
PRINT_EVERY = max(1, EPOCHS // 20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

# ───────────────────────────── Network definition ────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.Tanh(),
            nn.Linear(256, 512), nn.Tanh(),
            nn.Linear(512, 512), nn.Tanh(),
            nn.Linear(512, 512), nn.Tanh(),
            nn.Linear(512, 512), nn.Tanh(),
            nn.Linear(512, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

model = MLP(2 + N_EPS).to(device)
optimizer = Adam(model.parameters(), lr=LR)

# ───────────────────────────── Sampling helpers ──────────────────────────────

def sample_uniform(shape, low=0.0, high=1.0):
    return (high - low) * torch.rand(*shape, device=device) + low


def sample_xt(batch):
    x = sample_uniform((batch, 1))
    t = sample_uniform((batch, 1))
    return x, t


def sample_eps(batch):
    return sample_uniform((batch, N_EPS), -EPS_RANGE, EPS_RANGE)

f_ic = lambda x: torch.sin(torch.pi * x)

# ───────────────────────────── Derivative helpers ────────────────────────────

def derivatives(u, x, t):
    u_x, u_t = torch.autograd.grad(u, (x, t), grad_outputs=torch.ones_like(u), create_graph=True)
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_t, u_x, u_xx

# ───────────────────────────── Loss components ───────────────────────────────

def pde_residual():
    x, t = sample_xt(BATCH_PDE)
    eps = sample_eps(BATCH_PDE)                  # random ε
    x.requires_grad_(True); t.requires_grad_(True); eps.requires_grad_(True)
    u = model(torch.cat((x, t, eps), dim=1))
    u_t, _, u_xx = derivatives(u, x, t)
    return (u_t - u_xx).pow(2).mean()


def ic_residual():
    """Initial‑condition loss computed **only at ε = 0**."""
    x = sample_uniform((BATCH_IC, 1))
    eps_zero = torch.zeros((BATCH_IC, N_EPS), device=device)
    u = model(torch.cat((x, torch.zeros_like(x), eps_zero), dim=1))
    return (u - f_ic(x)).pow(2).mean()


def bc_residual():
    """Boundary loss enforced for *all* ε values (IC still ε=0 only)."""
    t = sample_uniform((BATCH_BC, 1))
    eps = sample_eps(BATCH_BC)                           # random ε like PDE
    zeros = torch.zeros_like(t)
    ones  = torch.ones_like(t)
    u_left  = model(torch.cat((zeros, t, eps), dim=1))
    u_right = model(torch.cat((ones , t, eps), dim=1))
    return u_left.pow(2).mean() + u_right.pow(2).mean()


def ortho_residual():
    x, t = sample_xt(BATCH_ORTHO)
    eps = sample_eps(BATCH_ORTHO); eps.requires_grad_(True)
    u = model(torch.cat((x, t, eps), dim=1))
    grads = [torch.autograd.grad(u, eps, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0][:, i:i+1]
             for i in range(N_EPS)]
    U = torch.cat(grads, dim=1)
    G = (U.T @ U) / BATCH_ORTHO
    off_diag = G - torch.diag(torch.diagonal(G))
    return off_diag.sum().pow(2)

# ───────────────────────────── Training loop ────────────────────────────────
for epoch in tqdm(range(1, EPOCHS + 1), desc="training"):
    optimizer.zero_grad(set_to_none=True)
    loss = (LAMBDA_PDE   * pde_residual() +
            LAMBDA_IC    * ic_residual()  +
            LAMBDA_BC    * bc_residual()  +
            LAMBDA_ORTHO * ortho_residual())
    loss.backward(); optimizer.step()
    if epoch % PRINT_EVERY == 0 or epoch == 1:
        tqdm.write(f"epoch {epoch:6d} | loss {loss.item():.6e}")
print("✓ Training finished")

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
    html.H1("Lie-PINN Solution u(x, t, ε)"),
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
    [Input(f'eps-slider-{i}', 'value') for i in range(N_EPS)] # Corrected Input list
)
def update_graph(*eps_values):
    """This function is called whenever any slider value changes."""
    # eps_values will be a tuple of the current slider values
    u_surface = compute_surface(eps_values)

    # Create the figure dynamically
    fig = go.Figure(data=[
        go.Surface(
            z=u_surface,
            x=xs_np,
            y=ts_np,
            colorscale="Viridis",
            cmin=-0.1, # Optional: fix colorscale range
            cmax=1.1,
            showscale=False
        )
    ])

    # Apply layout settings
    fig.update_layout(
        # title="Surface changes dynamically with sliders", # Title in H1 now
        scene=dict(
            xaxis_title="x",
            yaxis_title="t",
            zaxis=dict(title="u", range=[-0.1, 1.2]), # Adjust range as needed
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1)
        ),
        margin=dict(l=0, r=0, b=0, t=0) # Minimal margins
    )
    return fig

# Run the app
if __name__ == '__main__':
    # Make sure model is on the correct device after training
    model.to(device)
    print(f"Starting Dash server... Access at http://127.0.0.1:8050/ (Press CTRL+C to quit)")
    # Turn off reloader if running in a notebook or environment where it causes issues
    app.run(debug=True, use_reloader=False) # Use debug=False for production