import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

# -------------------- CONFIGURATION --------------------
plot_dir = Path("plots/m4")
plot_dir.mkdir(parents=True, exist_ok=True)

NX, NY = 60, 40
NSTEPS = 1000
SAVE_EVERY = 100
OMEGA_VALUES = [1.0, 1.2, 1.4, 1.6]  # Within (0,2)
u0 = 0.08  # initial amplitude (|u| < 0.1)
rho0 = 1.0
n = 1  # wave mode
PLOT_FLAG = False  # Set to True to enable plotting

E = torch.tensor([
    [0,  0], [1,  0], [0,  1], [-1,  0], [0, -1],
    [1,  1], [-1,  1], [-1, -1], [1, -1]
], dtype=torch.float32)

W = torch.tensor([
    4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9,
    1 / 36, 1 / 36, 1 / 36, 1 / 36
], dtype=torch.float32)


def feq(rho, u):
    """Compute the equilibrium distribution function."""
    eu = torch.einsum('xyi,ji->xyj', u, E)
    u2 = (u ** 2).sum(dim=2, keepdim=True)
    feq = rho.unsqueeze(-1) * W * (1 + 3 * eu + 4.5 * eu ** 2 - 1.5 * u2)
    return feq


def collide(f, omega):
    """Perform the collision step."""
    rho = f.sum(dim=2)
    u = compute_velocity(f, rho)
    feq_ = feq(rho, u)
    return f - omega * (f - feq_)


def stream(f):
    """Perform the streaming step."""
    f_new = torch.empty_like(f)
    for i in range(9):
        dx, dy = E[i].int()
        f_new[:, :, i] = torch.roll(f[:, :, i], shifts=(dx.item(), dy.item()), dims=(0, 1))
    return f_new


def compute_velocity(f, rho):
    """Compute the macroscopic velocity field."""
    momentum = torch.einsum('xyj,jk->xyk', f, E)
    u = momentum / rho.unsqueeze(-1)
    return u


def init_state(omega):
    """Initialize the simulation with sinusoidal velocity in x-direction."""
    Y = torch.arange(NY).view(1, NY).expand(NX, NY)
    k = 2 * math.pi * n / NY
    u = torch.zeros((NX, NY, 2))
    u[:, :, 0] = u0 * torch.sin(k * Y)
    rho = torch.full((NX, NY), rho0)
    f = feq(rho, u)
    return f, rho, u


def save_snapshot(u, rho, step, tag):
    """Save density and velocity plots."""
    (plot_dir / tag / "velocity").mkdir(parents=True, exist_ok=True)
    (plot_dir / tag / "density").mkdir(parents=True, exist_ok=True)
    X, Y = torch.meshgrid(torch.arange(NX), torch.arange(NY), indexing='ij')
    u_np = u.numpy()
    rho_np = rho.numpy()

    # Velocity field
    plt.figure()
    # speed = np.linalg.norm(u, axis=-1)
    plt.quiver(X, Y, u[..., 0], u[..., 1], scale=1.0)
    # plt.colorbar(label="Velocity Magnitude")
    plt.title(f"Velocity Field at Step {step} (omega = {omega})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(plot_dir / tag / "velocity" / f"velocity_{step:04d}.png")
    plt.close()

    # Density field
    plt.figure()
    plt.imshow(rho_np.T, origin='lower', cmap='viridis')
    plt.title(f"Density Field - Step {step} (omega = {omega})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="Density")
    plt.savefig(plot_dir / tag / "density" / f"density_{step:04d}.png")
    plt.close()


def run_simulation(omega):
    """Run LBM simulation for a given omega and return amplitude decay list."""
    tag = f"omega_{omega:.2f}"
    f, _, _ = init_state(omega)
    k = 2 * math.pi * n / NY
    amp_decay = []

    for step in range(NSTEPS + 1):
        rho = f.sum(dim=2)
        u = compute_velocity(f, rho)

        if step % SAVE_EVERY == 0 and PLOT_FLAG:
            save_snapshot(u, rho, step, tag)
        amp = u[NX // 2, NY // 4, 0].item()
        amp_decay.append(amp)

        f = collide(f, omega)
        f = stream(f)

    return amp_decay, tag


def plot_amplitude_decay(amp_list, omega):
    """Plot amplitude decay over time."""
    steps = np.arange(len(amp_list))
    amps = np.array(amp_list)
    plt.plot(steps, amps / amps[0], label=f"omega={omega:.2f}")


def extract_viscosity_numerical(amp_list):
    """Fit an exponential decay and extract effective viscosity."""
    y = np.log(np.array(amp_list) / amp_list[0])
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    decay_rate = -slope
    ky = (2 * math.pi * n / NY)
    nu = decay_rate / (ky ** 2)
    return nu


def plot_viscosity_vs_omega(omega_vals, nu_numeric):
    """Plot viscosity vs omega: numerical vs analytical."""
    omega_vals = np.array(omega_vals)
    nu_analytical = (1 / omega_vals - 0.5) / 3

    plt.figure()
    plt.plot(omega_vals, nu_numeric, 'bo-', label='Numerical')
    plt.plot(omega_vals, nu_analytical, 'r--', label='Analytical')
    plt.xlabel("Omega")
    plt.ylabel("Kinematic Viscosity")
    plt.title("Viscosity vs Omega")
    plt.legend()
    plt.grid()
    plt.savefig(plot_dir / "viscosity_vs_omega.png")
    plt.close()

def print_viscosity_table(omega_vals, nu_numeric):
    nu_analytical = [(1 / omega - 0.5) / 3 for omega in omega_vals]
    print("\nOmega | Numerical ν | Analytical ν | Abs Error")
    print("------|--------------|----------------|-----------")
    for o, n, a in zip(omega_vals, nu_numeric, nu_analytical):
        print(f"{o:.2f}   | {n:.6f}     | {a:.6f}        | {abs(n-a):.6f}")

def find_best_omega(omega_vals, nu_numeric):
    """Return the omega with smallest viscosity error."""
    omega_vals = np.array(omega_vals)
    nu_numeric = np.array(nu_numeric)
    nu_analytical = (1 / omega_vals - 0.5) / 3
    errors = np.abs(nu_numeric - nu_analytical)

    best_idx = np.argmin(errors)
    best_omega = omega_vals[best_idx]
    best_nu_num = nu_numeric[best_idx]
    best_nu_ana = nu_analytical[best_idx]
    best_error = errors[best_idx]

    print("\n--- Best Omega Summary ---")
    print(f"Best Omega        : {best_omega:.3f}")
    print(f"Numerical Viscosity : {best_nu_num:.6f}")
    print(f"Analytical Viscosity: {best_nu_ana:.6f}")
    print(f"Absolute Error      : {best_error:.6f}")
    return best_omega, best_nu_num, best_nu_ana, best_error

if __name__ == "__main__":
    viscosity_measurements = []
    for omega in OMEGA_VALUES:
        print(f"Running simulation for omega = {omega:.2f}")
        amp_decay, tag = run_simulation(omega)
        plot_amplitude_decay(amp_decay, omega)
        viscosity_measurements.append(extract_viscosity_numerical(amp_decay))

    plt.title("Amplitude Decay")
    plt.xlabel("Time step")
    plt.ylabel("Normalized Amplitude")
    plt.legend()
    plt.savefig(plot_dir / "amplitude_decay_all.png")
    plt.close()

    plot_viscosity_vs_omega(OMEGA_VALUES, viscosity_measurements)
    print_viscosity_table(OMEGA_VALUES, viscosity_measurements)
    best_omega, best_nu_num, best_nu_ana, best_error = find_best_omega(OMEGA_VALUES, viscosity_measurements)