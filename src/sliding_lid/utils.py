import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum
from constants import *

class InitMode(IntEnum):
    UNIFORM = 1
    TEST = 2


def theoretical_viscosity(omega: float = 1.) -> float:
    """
    Analytical calculation of viscosity.
    Args:
        omega (float): collision frequency (inverse of relaxation time).
    Returns:
        Theoretical fluid viscosity value.
    """
    return (1/omega - 0.5) / 3


def theoretical_decay_calcs(initial_amp: float,
                            omega: float,
                            length: float,
                            timestep: list
) -> torch.Tensor:
    """
    Calculate the theoretical decay.
        initial_amp (float): Sinusoidal wave amplitude in the initial condition
        omega (float): collision frequency (inverse of relaxation time).
        length (float): lattice length on the y-coordinates
        timestep (float): timestep
    Returns:
        Theoretically calculated decay.

    """
    # Calculate viscosity
    viscosity = theoretical_viscosity(omega)

    # Theoretical calculation of the exponential decay using torch
    time_tensor = torch.tensor(timestep, dtype=torch.float32, device=DEVICE)
    decay = -viscosity * (2 * np.pi / length) ** 2 * time_tensor
    return initial_amp * torch.exp(decay)

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
