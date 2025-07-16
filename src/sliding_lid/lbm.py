import torch
import os
import numpy as np
from tqdm import trange
from typing import Optional
import matplotlib.pyplot as plt
from constants import *
from utils import InitMode

if PLOT_FLAG:
    PLOT_DIR = Path("plots/lbm/")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def init_distribution(
    nx: Optional[int] = 100,
    ny: Optional[int] = 100,
    n_dir: Optional[int] = 9,
    init_state: Optional[int] = InitMode.UNIFORM,
):
    """
    Initialize the probability density function.
    Args:
        nx : Width of lattice
        ny : Height of lattice
        n_dir : Number of channels or directions in the lattice
        init_state : Initial state of the lattice
    Returns:
        f = Probability density function for all channels and every point in the lattice
    """
    # Initialize the probability density function with zeros
    probab_density_f = torch.zeros((n_dir, nx, ny), dtype=torch.float32, device=DEVICE)

    # Uniform initialization of the probability density function
    if init_state == InitMode.UNIFORM:
        for i in range(nx):
            for j in range(ny):
                probab_density_f[:, i, j] = WEIGHTS

    # init_state initialized with 1 on the center position
    if init_state == InitMode.TEST:
        probab_density_f[:, int(nx // 2), int(ny // 2)] = 1
        return torch.tensor(probab_density_f, dtype=torch.float32, device=DEVICE)

    return torch.tensor(probab_density_f, dtype=torch.float32, device=DEVICE)

def compute_density(probab_density_f):
    """
    Compute the density at each given lattice point
    """
    # Sum over the channels to get the density at each point
    return torch.sum(probab_density_f, dim=0)

def compute_velocity(probab_density_f):
    """
    Compute the velocity field at each given point.
    Args:
        probab_density_f : Probability density function
    Returns:
        The velocity field in shape (2, X, Y): at each point the vector gives the average velocity in the x and y direction
    """

    density = compute_density(probab_density_f)
    velocity = torch.matmul(
        CHANNEL_VELOCITIES.T.to(probab_density_f.dtype).to(probab_density_f.device),
        probab_density_f.reshape(9, -1),
    ).reshape(2, *density.shape)
    return velocity / density

def streaming(probab_density_f):
    """
    Perform the streaming step of the Lattice Boltzmann method.
    Shifts the distribution functions in their respective directions using periodic boundaries.
    """
    # n_dir = CHANNEL_VELOCITIES.shape[0]
    # for i in range(n_dir):
    #     probab_density_f[i] = torch.roll(
    #         probab_density_f[i],
    #         shifts=(int(CHANNEL_VELOCITIES[i][0]), int(CHANNEL_VELOCITIES[i][1])),
    #         dims=(0, 1),
    #     )
    """Fully vectorized streaming - much faster"""
    # Pre-compute all shifts to avoid Python loops
    shifts = torch.tensor([
        [int(CHANNEL_VELOCITIES[i][0]), int(CHANNEL_VELOCITIES[i][1])] 
        for i in range(9)
    ], device=DEVICE)
    
    # Vectorized streaming using stack
    probab_density_f[:] = torch.stack([
        torch.roll(probab_density_f[i], 
                  shifts=tuple(shifts[i].tolist()), 
                  dims=(0, 1))
        for i in range(9)
    ])

def compute_equilibrium(rho, u):
    """
    Calculate the equilibrium distribution given the density(rho) and average velocity.
    Args:
        rho : Mass density at each position of the grid
        u : Average velocity at each position of the grid
    Returns:
        Equilibrium distribution at each (x, y) point of the grid.
    """
    v_flat = u.reshape(2, -1)
    temp_v = torch.matmul(
        CHANNEL_VELOCITIES.to(v_flat.dtype).to(v_flat.device), v_flat
    ).reshape(9, *u.shape[1:])
    temp_v_squared = torch.norm(u, dim=0) ** 2
    result = WEIGHTS[:, None, None] * (
        rho * (1 + 3 * temp_v + 4.5 * temp_v**2 - 1.5 * temp_v_squared)
    )
    return result

def collision_relaxation(probab_density_f, velocity, rho, omega: Optional[float] = 0.5):
    """
    Calculate the collision operation.
    Args:
        probab_density_f : Probability density function
        velocity : Average velocity at each position of the grid of shape
        rho : Mass density at each position of the grid of shape
        omega (Optional[float]): The collision frequency. Default value is 0.5
    Returns:
        The probability density function at each point in the grid after the
        streaming and collision operations are applied.
    """
    f_eq = compute_equilibrium(rho, velocity)
    probab_density_f += omega * (f_eq - probab_density_f)


def plot_density(rho, step) -> None:
    """
    Create a heatmap of the density at each lattice point.

    Args:
        density : Mass density at each position of the grid.
        step : Simulation step for labeling the plot.
    Returns:
        None
    """
    plt.imshow(rho.T, origin="lower", cmap="viridis")
    plt.colorbar(label="Density")
    plt.xlabel("Lattice X Position")
    plt.ylabel("Lattice Y Position")
    plt.title(f"Density at Step {step}")
    plt.savefig(PLOT_DIR / f"density_step_{step:04d}.png")
    plt.close()


def plot_velocity_field(u, step, nx, ny) -> None:
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')

        plt.figure(figsize=(6, 4))
        plt.quiver(X, Y, u[0].T, u[1].T, scale=1, scale_units='xy')
        plt.xlabel("Lattice X Position")
        plt.ylabel("Lattice Y Position")
        plt.title(f"Velocity Field at Step {step}")
        plt.savefig(PLOT_DIR / f"velocity_step_{step:04d}.png")
        plt.close()


def rigid_wall(
    probab_density_f: torch.Tensor,
    pre_streaming_probab_density: torch.Tensor,
    location: Optional[str] = "lower",
) -> None:
    """
    Apply rigid wall boundary conditions.
    Args:
        probab_density_f : Probability density function of Lattice.
        pre_streaming_probab_density : Probability density function before the streaming operator is applied
        location (Optional[str]): Physical location of the boundary
    Returns:
         None.
    """

    # Lower wall
    if location == "lower":
        idx = 0
        out_channels = down_out_channels
        in_channels = down_in_channels

    # Upper wall
    elif location == "upper":
        idx = -1
        out_channels = up_out_channels
        in_channels = up_in_channels

    # Right wall
    elif location == "right":
        idx = -1
        out_channels = right_out_channels
        in_channels = right_in_channels

    # Left wall
    elif location == "left":
        idx = 0
        out_channels = left_out_channels
        in_channels = left_in_channels

    else:
        raise ValueError("Invalid location given: '" + location)

    if location in ("upper", "lower"):
        # Loop over channels and apply boundary conditions
        for i in range(len(in_channels)):
            # Set temporary variables for convenience
            temp_in, temp_out = in_channels[i], out_channels[i]
            # Index of y's that are on the lower boundary is 0
            probab_density_f[temp_in, :, idx] = pre_streaming_probab_density[
                temp_out, :, idx
            ]

    elif location in ("right", "left"):
        for i in range(len(in_channels)):
            # Set temporary variables for convenience
            temp_in, temp_out = in_channels[i], out_channels[i]

            probab_density_f[temp_in, idx, :] = pre_streaming_probab_density[
                temp_out, idx, :
            ]

    else:
        raise ValueError("Invalid location provided")

def moving_wall(
    probab_density_f: torch.Tensor,
    pre_streaming_probab_density: torch.Tensor,
    wall_velocity: torch.Tensor,
    density: torch.Tensor,
    location: Optional[str] = "bottom",
) -> None:
    """
    Apply moving wall boundary conditions.
    Args:
        probab_density_f : Probability density function
        pre_streaming_probab_density : Probability density function before the streaming operator is applied
        wall_velocity : Velocity of the moving wall as a vector
        density : Mass density at each position of the grid
        location (Optional[str]): Physical location of the boundary

    Returns:
        None.
    """
    # Calculate average density
    avg_density = density.mean()

    if location == "upper":
        # Channels going out
        out_channels = up_out_channels
        # Channels going in
        in_channels = up_in_channels
        # Loop over channels and apply boundary conditions
        for i in range(len(in_channels)):
            # Set temporary variables for convenience
            temp_in, temp_out = in_channels[i], out_channels[i]
            # Calculate term due to velocity based on the channels going out
            temp_term = (
                -2 * WEIGHTS[temp_out] * avg_density / c_s_squared
            ) * torch.dot(
                CHANNEL_VELOCITIES[temp_out].to(wall_velocity.dtype), wall_velocity
            )
            # Index of y's that are on the upper boundary is equal to the
            # size of the lattice - 1, for simplicity use "-1" to access
            probab_density_f[temp_in, :, -1] = (
                pre_streaming_probab_density[temp_out, :, -1] + temp_term
            )
    else:
        raise ValueError("Invalid location given: '" + location)


if __name__ == "__main__":
    nx = 15
    ny = 10
    f = init_distribution(nx, ny, init_state=InitMode.UNIFORM)
    rho = torch.empty((nx, ny), dtype=torch.float32, device=DEVICE)
    v = torch.empty((2, nx, ny), dtype=torch.float32, device=DEVICE)
    compute_density(f, out=rho)
    compute_velocity(f, out=v)

    # Increase the mass at a somewhat central point in the grid
    f[:, nx // 2, ny // 2] += 0.01 * f[:, nx // 2, ny // 2 - 2]

    # Calculate streaming part and plot density for 10 timesteps
    for step in range(20):
        streaming(f)
        compute_density(f, out=rho)
        compute_velocity(f, out=v)
        if PLOT_FLAG:
            plot_density(rho, step)
            plot_velocity_field(v, step, nx, ny)
