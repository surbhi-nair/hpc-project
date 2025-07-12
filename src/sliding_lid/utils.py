"""Utilities and side functions for main calculations"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from enum import IntEnum
from globals import c_s_squared


class InitMode(IntEnum):
    DEV = 1
    RAND = 2
    EQUILIBRIUM = 3


def theoretical_viscosity(omega: float = 1.) -> float:
    """
    Analytical calculation of viscosity.
    Args:
        omega (float): collision frequency (inverse of relaxation time).
    Returns:
        Fluid viscosity.
    """
    return (1/omega - 0.5) / 3


def theoretical_decay_calcs(initial_amp: float,
                            omega: float,
                            length: float,
                            time: list
) -> torch.Tensor:
    """
    Calculate the theoretical decay.
        initial_amp (float): Sinusoidal wave amplitude in the initial condition
        omega (float): collision frequency (inverse of relaxation time).
        length (float): lattice length on the y-coordinates
        time (float): timestep
    Returns:
        Theoretically calculated decay.

    """
    # Calculate viscosity
    viscosity = theoretical_viscosity(omega)

    # Theoretical calculation of the exponential decay using torch
    time_tensor = torch.tensor(time, dtype=torch.float32, device=DEVICE)
    decay = -viscosity * (2 * np.pi / length) ** 2 * time_tensor
    return initial_amp * torch.exp(decay)


def theoretical_poiseuille_flow(
        density,
        omega: float,
        density_input: float,
        density_output: float
) -> torch.Tensor:
    """
    Calculate the analytical solution of the velocity field for a
        Hagen-Poiseuille flow in a pipe.

    Args:
        density (np.ndarray): Mass density at each position of the grid.
        omega (float): Collision frequency
        density_input (float): Density value at the input.
        density_output (float): Density value at the output.

    Returns:
        The velocity field as a np.ndarray.
    """
    # Get dimensions
    X, Y = density.shape

    # Create points to be returned
    points = torch.linspace(0, Y, Y, device=DEVICE)

    # Calculate the mean value of density using torch
    mean_density = torch.mean(torch.tensor(density, dtype=torch.float32, device=DEVICE)) if isinstance(density, np.ndarray) else torch.mean(density)
    # Calculate viscosity
    viscosity = theoretical_viscosity(omega)
    # Calculate dynamic viscosity
    dynamic_viscosity = mean_density * viscosity
    # Calculate derivative nominator as the difference in derivatives
    derivative_nom = density_output - density_input

    # Calculate velocity field
    return -(derivative_nom * points * (Y - points)) / (2 * X * dynamic_viscosity) * c_s_squared


def save_streamplot(velocity,
                    step: int,
                    ax: plt.Axes, ) -> None:
    """
    Save intermediate velocity streamplot to generate gif of velocity field over time.

    Args:
        velocity (np.ndarray): Average velocity at each position of the grid of shape
            (2, X, Y).
        step (int): Simulation step
        ax (plt.Axes): Matplotlib axes element for the streamplot.

    Returns:
        None
    """
    # Clear axes
    ax.clear()

    # Get dimensions
    X, Y = velocity.shape[1:]

    x, y = np.meshgrid(range(X), range(Y))

    velocity_cpu = velocity.detach().cpu().numpy() if torch.is_tensor(velocity) else velocity
    velocity_x = np.moveaxis(velocity_cpu[0], 0, 1)
    velocity_y = np.moveaxis(velocity_cpu[1], 0, 1)

    # Generate streamplot with varying colors
    streamplot = ax.streamplot(x, y, velocity_x, velocity_y)

    # Save plot
    path = "data/sliding_lid_images"
    path_exists = os.path.exists(path)
    if not path_exists:
        # Create path if it does not exist
        os.makedirs(path)
    plt.savefig(path + f'/sliding_lid_velocity_field_{step:04d}')
