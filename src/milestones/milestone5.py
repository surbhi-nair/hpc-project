import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import time

torch.set_float32_matmul_precision('high')  # Enable TF32 for better performance
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Milestone 5 - Using device:", DEVICE)

# Constants for D2Q9
E = torch.tensor([
    [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1]
], dtype=torch.float32, device=DEVICE)

W = torch.tensor([
    4/9, 1/9, 1/9, 1/9, 1/9,
    1/36, 1/36, 1/36, 1/36
], dtype=torch.float32, device=DEVICE)

OPP = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], device=DEVICE)

# Simulation parameters
NX, NY = 3000, 3000
NSTEPS = 10000
OMEGA = 1.0
TAU = 1 / OMEGA
LID_VELOCITY = 0.1

PLOT_FLAG = False # Set to True to enable plotting
PLOT_DIR = ""
if PLOT_FLAG:
    PLOT_DIR = Path("plots/milestones/m5")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


# Save frequency for plots and velocity snapshots
SAVE_EVERY = 100

def equilibrium(rho, u):
    """Compute the local equilibrium distribution function f_eq."""
    cu = torch.einsum('xyi,ji->xyj', u, E)
    usqr = (u ** 2).sum(-1, keepdim=True)
    feq = rho.unsqueeze(-1) * W * (1 + 3 * cu + 4.5 * cu ** 2 - 1.5 * usqr)
    return feq

def initialize():
    """Initialize rho=1, u=0 and compute initial f from equilibrium."""
    rho = torch.ones((NX, NY), dtype=torch.float32, device=DEVICE)
    u = torch.zeros((NX, NY, 2), dtype=torch.float32, device=DEVICE)
    f = equilibrium(rho, u)
    return rho, u, f

def compute_macroscopic(f):
    """Calculate density and velocity from f."""
    rho = f.sum(dim=-1)
    momentum = torch.einsum('xyi,ij->xyj', f, E)
    u = momentum / rho.unsqueeze(-1)
    return rho, u

@torch.compile
def stream_and_apply_boundaries(f):
    """Perform streaming and apply boundary conditions fused into one function."""
    f_streamed = torch.empty_like(f)
    for i in range(9):
        dx, dy = int(E[i, 0].item()), int(E[i, 1].item())
        f_streamed[..., i] = torch.roll(f[..., i], shifts=(dx, dy), dims=(0, 1))

    rho = f.sum(dim=-1)

    # Bottom wall (y=0): bounce-back for directions 2,5,6
    for i in [2, 5, 6]:
        opp = OPP[i].item()
        f_streamed[:, 0, i] = f[:, 0, opp]
    # Left wall (x=0): bounce-back for directions 1,5,8
    for i in [1, 5, 8]:
        opp = OPP[i].item()
        f_streamed[0, :, i] = f[0, :, opp]
    # Right wall (x=NX-1): bounce-back for directions 3,6,7
    for i in [3, 6, 7]:
        opp = OPP[i].item()
        f_streamed[-1, :, i] = f[-1, :, opp]
    # Moving lid (y=NY-1): Zou-He moving wall for directions 4,7,8
    u_wall = torch.tensor([LID_VELOCITY, 0], dtype=torch.float32, device=DEVICE)
    rho_top = rho[:, -1].detach()
    ci_list = [E[i] for i in [4, 7, 8]]
    wi_list = [W[i] for i in [4, 7, 8]]
    opp_list = [OPP[i].item() for i in [4, 7, 8]]

    for idx, i in enumerate([4, 7, 8]):
        ci = ci_list[idx]
        ci_dot_u = ci[0] * u_wall[0] + ci[1] * u_wall[1]
        wi = wi_list[idx]
        opp = opp_list[idx]
        f_eq_term = wi * rho_top * (3 * ci_dot_u)
        f_streamed[:, -1, i] = f[:, -1, opp] + 2 * f_eq_term

    return f_streamed

@torch.compile
def collide(f):
    """Collision step using BGK approximation."""
    rho, u = compute_macroscopic(f)
    feq = equilibrium(rho, u)
    f += -1 / TAU * (f - feq)
    return f

def save_velocity_plot(u, step):
    """Plot the velocity vector field using quiver."""
    u_np = u.cpu().numpy()
    X, Y = np.meshgrid(np.arange(NX), np.arange(NY), indexing='ij')
    plt.figure(figsize=(6, 6))
    plt.quiver(X, Y, u_np[..., 0], u_np[..., 1], scale=3, scale_units='xy')
    plt.title(f"Velocity Field at Step {step}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(PLOT_DIR / f"velocity_step_{step:04d}.png")
    plt.close()
    
def save_streamplot(u, step):
    """Save a streamline plot of velocity vectors at a given step."""
    u_np = u.detach().cpu().numpy()  # Shape: (NX, NY, 2)

    Y, X = np.meshgrid(np.arange(NY), np.arange(NX), indexing='ij')  # Shape: (NY, NX)
    U = u_np[..., 0]  # Shape: (NX, NY)
    V = u_np[..., 1]  # Shape: (NX, NY)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.streamplot(X, Y, U.T, V.T, density=1.2, linewidth=0.8, arrowsize=1)
    ax.set_title(f"Streamplot at Step {step}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    streamplot_dir = PLOT_DIR / "streamplots"
    streamplot_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(streamplot_dir / f'sliding_lid_velocity_field_{step:04d}.png')
    plt.close(fig)
def run_simulation():
    """Main simulation loop."""
    rho, u, f = initialize()
    # Dictionary to store the x-component of velocity along the vertical centerline at intervals
    vx_dict = {}
    center_x = NX // 2
    start = time.time()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    with torch.no_grad():
        for step in range(NSTEPS):
            # pre_f = f.detach().clone()            # Save pre-streaming distributions without tracking gradients
            f = stream_and_apply_boundaries(f)   # Streaming and BC fused
            f = collide(f)

            # Save plots and velocity slices at specified intervals
            if (step % SAVE_EVERY == 0 or step == NSTEPS - 1) and PLOT_FLAG:
                _, u = compute_macroscopic(f)
                #save_velocity_plot(u, step)
                save_streamplot(u, step)
                # Store the x-component of velocity along the vertical centerline (y-direction)
                # Shape: (NY,)
                vx_dict[step] = u[center_x, :, 0].detach().cpu().numpy()

    end_event.record()
    torch.cuda.synchronize()  # Wait for GPU to finish
    gpu_time_sec = start_event.elapsed_time(end_event) / 1000  # ms -> s
    total_updates = NSTEPS * NX * NY
    blups = total_updates / gpu_time_sec / 1e9
    
    print(f"========= Performance: {blups:.3f} BLUPS (GPU Time: {gpu_time_sec:.3f} s) =========")
    mlups = total_updates / gpu_time_sec / 1e6
    print(f"========= Performance: {mlups:.3f} MLUPS (GPU Time: {gpu_time_sec:.3f} s) =========")

    end = time.time()
    T = end - start
    updates = NSTEPS * NX * NY
    blups = updates / T / 1e9
    print(f"Performance: {blups:.3f} billion lattice updates per second (BLUPS)")
    mlups = updates / T / 1e6
    print(f"Performance: {mlups:.3f} million lattice updates per second (MLUPS)")

    # Optionally: save vx_dict for later analysis/visualization
    if PLOT_FLAG:
        np.save(PLOT_DIR / "vx_centerline_dict.npy", vx_dict)

if __name__ == "__main__":
    run_simulation()