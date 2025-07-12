"""Lattice Boltzmann Method core functions"""
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from tqdm import trange
from typing import Optional
from globals import *
from utils import InitMode


def init_proba_density(width: Optional[int] = 15,
                       height: Optional[int] = 10,
                       n_channels: Optional[int] = 9,
                       mode: Optional[int] = InitMode.EQUILIBRIUM,
                       seed: Optional[bool] = False):
    """
    Initialize the probability density function.
    Args:
        width (Optional[int]): The width of the lattice. Default 15.
        height (Optional[int]): The height of the lattice. Default 10.
        n_channels (Optional[int]): Number of channels of velocity discretization. Default 9.
        mode (Optional[int]): Parameter determining the initialization mode. Default EQUILIBRIUM.
        seed (Optional[bool]): Boolean flag for seed in random initialization. Used for testing.
            Default False.
    Returns:
        Probability density function for all channels and every point in the
            lattice.
        """
    # Initialization
    proba_density = np.zeros((n_channels, width, height))

    # Development mode initialized with one on the center position
    if mode == InitMode.DEV:
        proba_density[:, int(width // 2), int(height // 2)] = 1
        return torch.tensor(proba_density, dtype=torch.float32, device=DEVICE)

    # Equilibrium occupation numbers, initialized with equilibrium weights
    elif mode == InitMode.EQUILIBRIUM:
        for i in range(width):
            for j in range(height):
                proba_density[:, i, j] = weights

    # Initialize with random numbers
    elif mode == InitMode.RAND:
        if seed:
            np.random.seed(42)

        proba_density = np.random.rand(n_channels * width * height).reshape((n_channels, width, height))
    return torch.tensor(proba_density, dtype=torch.float32, device=DEVICE)


def calculate_density(proba_density):
    """
    Calculate the mass density at each given point.
    Args:
        proba_density (np.ndarray): Probability density function of Lattice Boltzmann
            Equation.
    Returns:
        density (np.ndarray): Mass density at each position of the grid of shape
            (X, Y)
    """
    # Sum over the different channels of probability density function
    return torch.sum(proba_density, dim=0)


def calculate_velocity(proba_density):
    """
    Calculate the velocity field at each given point.
    Args:
        proba_density (np.ndarray): Probability density function of Lattice Boltzmann
            Equation.
    Returns:
        The velocity field as a numpy array of shape (2, X, Y).
        At each point in real space we get a vector depicting the average velocity
        at the x- and y- direction.
    """

    density = calculate_density(proba_density)
    # velocity_channels: (9, 2), proba_density: (9, X, Y)
    # want velocity: (2, X, Y)
    velocity = torch.matmul(
        velocity_channels.T.to(proba_density.dtype).to(proba_density.device),
        proba_density.reshape(9, -1)
    ).reshape(2, *density.shape)
    return velocity / density


def streaming(proba_density):
    """
    Calculate the L.H.S. streaming operation of the Lattice Boltzmann equation
    by shifting the components of the probability density function along a grid.
    Args:
        proba_density (np.ndarray): Probability density function of Lattice Boltzmann
            Equation.
    Returns:
        None
    """

    n_channels = velocity_channels.shape[0]
    for i in range(n_channels):
        proba_density[i] = torch.roll(
            proba_density[i],
            shifts=(int(velocity_channels[i][0]), int(velocity_channels[i][1])),
            dims=(0, 1)
        )


def calculate_equilibrium_distro(density, velocity):
    """
    Calculate the equilibrium distribution given the density and average
    velocity.
    Args:
        density (nd.array): Mass density at each position of the grid of shape
            (X, Y).
        velocity (nd.array): Average velocity at each position of the grid of shape
            (2, X, Y)
    Returns:
        Equilibrium distribution at each (x, y) point of the grid.
    """
    # velocity: (2, X, Y), velocity_channels: (9, 2)
    v_flat = velocity.reshape(2, -1)  # (2, X*Y)
    temp_v = torch.matmul(
        velocity_channels.to(v_flat.dtype).to(v_flat.device), v_flat
    ).reshape(9, *velocity.shape[1:])
    temp_v_squared = torch.norm(velocity, dim=0) ** 2
    # weights: (9,), density: (X, Y), temp_v: (9, X, Y), temp_v_squared: (X, Y)
    result = weights[:, None, None] * (density * (1 + 3 * temp_v + 4.5 * temp_v**2 - 1.5 * temp_v_squared))
    return result


def collision_relaxation(
        proba_density,
        velocity,
        density,
        omega: Optional[float] = 0.5):
    """
    Calculate the collision operation.
    Args:
        proba_density (np.ndarray): Probability density function of Lattice Boltzmann
        Equation.
        velocity (np.ndarray): Average velocity at each position of the grid of shape
            (2, X, Y)
        density (np.ndarray): Mass density at each position of the grid of shape
            (X, Y).
        omega (Optional[float]): The collision frequency. Default value is 0.5
    Returns:
        The probability density function at each point in the grid after the
        streaming and collision operations are applied.
    """
    eql_proba_density = calculate_equilibrium_distro(density, velocity)
    proba_density += omega * (eql_proba_density - proba_density)


def plot_density(density: np.array,
                 iter: Optional[int] = 0,
                 show: Optional[bool] = False) -> None:
    """
    Create a heatmap of the density at each lattice point.

    Args:
        density (np.ndarray): Mass density at each position of the grid.
        show (Optional[bool]): Whether to display the graphs or save them.
        iter (Optional[int]): Used to generate filename when saving the figures.
    Returns:
        None
    """
    # Convert to numpy if tensor
    if torch.is_tensor(density):
        density = density.cpu().numpy()
    # Calculate the labels of x- and y-axis
    width, height = density.shape

    column_labels = list(range(width))
    row_labels = list(range(height))

    fig, ax = plt.subplots()
    c = ax.pcolor(np.moveaxis(density, 0, 1), cmap=plt.cm.Reds)

    # put the major ticks at the middle of each cell
    _ = ax.set_xticks(np.arange(width) + 0.5, minor=False)
    _ = ax.set_yticks(np.arange(height) + 0.5, minor=False)

    _ = ax.invert_yaxis()

    _ = ax.set_xticklabels(column_labels, minor=False)
    _ = ax.set_yticklabels(row_labels, minor=False)
    _ = ax.set_ylabel('y-coordinate')
    _ = ax.set_xlabel('x-coordinate')
    _ = ax.set_title('Density')

    fig.colorbar(c, ax=ax)
    plt.grid()

    if show:
        plt.show()

    else:
        path_exists = os.path.exists("data")
        if not path_exists:
            # Create path if it does not exist
            os.makedirs("data")
        print("Saving plots under /data")
        plt.savefig('data/density_{}'.format(iter))
        plt.show()


def plot_velocity_field(velocity: np.array,
                        fig: plt.Figure,
                        ax: plt.Axes,
                        title: Optional[str] = "Velocity Field",
                        y_label: Optional[str] = "Y",
                        x_label: Optional[str] = "X") -> None:
    # Convert to numpy if tensor
    if torch.is_tensor(velocity):
        velocity = velocity.cpu().numpy()
    # Get dimensions
    X, Y = velocity.shape[1:]

    # Sample points periodically for clarity in the plot
    x_stride = max(X // 20, 1)
    y_stride = max(Y // 20, 1)
    x = np.arange(0, X, x_stride)
    y = np.arange(0, Y, y_stride)
    x_grid, y_grid = np.meshgrid(x, y)
    velocity_x = np.moveaxis(velocity[0], 0, 1)[::x_stride, ::y_stride]
    velocity_y = np.moveaxis(velocity[1], 0, 1)[::x_stride, ::y_stride]

    # Set ticks at sampled points only
    ax.set_xticks(x)
    ax.set_yticks(y)

    # Set tick labels matching ticks
    ax.set_xticklabels(x)
    ax.set_yticklabels(y)

    # Generate streamplot with color by velocity magnitude and improved colormap
    streamplot = ax.streamplot(
        x_grid, y_grid, velocity_x, velocity_y,
        color=np.sqrt(velocity_x**2 + velocity_y**2), cmap='viridis'
    )
    # Add colorbar
    fig.colorbar(streamplot.lines, ax=ax)

    # Add improved labels and title
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

def rigid_wall(
        proba_density: np.array,
        pre_streaming_proba_density: np.array,
        location: Optional[str] = "lower") -> None:
    """
    Apply rigid wall boundary conditions.
    Args:
        proba_density (np.ndarray): Probability density function of Lattice.
        pre_streaming_proba_density (np.ndarray): Probability density function before the
            streaming operator is applied
        location (Optional[str]): Physical location of the boundary.
            Currently, supports all boundary possibilities: ['left, 'right', lower', 'upper'].
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
        raise ValueError("Invalid location given: '" + location + "'. "
                         "Allowed values are: 'upper', 'lower', 'right', or 'left'.")

    if location in ("upper", "lower"):
        # Loop over channels and apply boundary conditions
        for i in range(len(in_channels)):
            # Set temporary variables for convenience
            temp_in, temp_out = in_channels[i], out_channels[i]
            # Index of y's that are on the lower boundary is 0
            proba_density[temp_in, :, idx] = \
                pre_streaming_proba_density[temp_out, :, idx]

    elif location in ("right", "left"):
        for i in range(len(in_channels)):
            # Set temporary variables for convenience
            temp_in, temp_out = in_channels[i], out_channels[i]

            proba_density[temp_in, idx, :] = \
                pre_streaming_proba_density[temp_out, idx, :]

    else:
        raise ValueError("Invalid location provided")


def moving_wall(
        proba_density: np.array,
        pre_streaming_proba_density: np.array,
        wall_velocity: np.array,
        density: np.array,
        location: Optional[str] = "bottom") -> None:
    """
    Apply moving wall boundary conditions.
    Args:
        proba_density (np.ndarray): Probability density function of Lattice.
        pre_streaming_proba_density (np.ndarray): Probability density function before the
            streaming operator is applied
        wall_velocity (np.ndarray): Velocity of the moving wall as a vector.
        density (np.ndarray): Mass density at each position of the grid.
        location (Optional[str]): Physical location of the boundary. For Couette flow only
            two possible positions: "upper" or "lower".

    Returns:
        None.
    """
    # Calculate average density
    avg_density = density.mean()

    if location == "lower":
        raise NotImplementedError("Not Implemented yet.")

    elif location == "upper":
        # Channels going out
        out_channels = up_out_channels
        # Channels going in
        in_channels = up_in_channels
        # Loop over channels and apply boundary conditions
        for i in range(len(in_channels)):
            # Set temporary variables for convenience
            temp_in, temp_out = in_channels[i], out_channels[i]
            # Calculate term due to velocity based on the channels going out
            temp_term = \
                (-2 * weights[temp_out] * avg_density / c_s_squared) * \
                np.dot(velocity_channels[temp_out], wall_velocity)
            # Index of y's that are on the upper boundary is equal to the
            # size of the lattice - 1, for simplicity use "-1" to access
            proba_density[temp_in, :, -1] = \
                pre_streaming_proba_density[temp_out, :, -1] + temp_term

    else:
        raise ValueError("Invalid location given: '" + location + "'. "
                         "Allowed values are: 'upper' or 'lower'.")


def pressure_gradient(
        proba_density: np.array,
        density: np.array,
        velocity: np.array,
        density_input: float,
        density_output: float,
        flow: Optional[str] = "left_to_right",
) -> None:
    """
    Apply periodic boundary conditions with pressure gradient.
    Args:
        proba_density (np.ndarray): Probability density function of Lattice.
        density (np.ndarray): Mass density at each position of the grid of shape (X, Y).
        velocity (np.ndarray): Velocity at each position of the grid of shape
            (2, X, Y).
        density_input (float): Density value at the input.
        density_output (float): Density value at the output.
        flow (Optional[str]): Denotes the direction of the flow. Currently, only
            left to right direction in x-axis is supported.

    Returns:
        None.
    """
    if flow == "left_to_right":
        # Calculate equilibrium distribution
        proba_equilibrium = calculate_equilibrium_distro(density, velocity)

        # Calculate equilibrium distribution using input density and output velocity
        equil_din_vout = calculate_equilibrium_distro(density_input, velocity[:, -2, :])

        # Calculate equilibrium distribution using output density and input velocity
        equil_dout_vin = calculate_equilibrium_distro(density_output, velocity[:, 1, :])

        # proba density at inlet
        proba_density[:, 0, :] = \
            equil_din_vout[:, :] + (proba_density[:, -2, :] - proba_equilibrium[:, -2, :])
        # proba density at outlet
        proba_density[:, -1, :] = \
            equil_dout_vin[:, :] + (proba_density[:, 1, :] - proba_equilibrium[:, 1, :])

    else:
        raise NotImplementedError("Flows other than left-to-right are not yet implemented.")


if __name__ == "__main__":
    # Initialize density function at equilibrium
    p_stream = init_proba_density(15, 15, mode=InitMode.EQUILIBRIUM)
    # Calculate density
    density = calculate_density(p_stream)
    # Calculate velocity
    v = calculate_velocity(p_stream)

    # Increase the mass at a somewhat central point in the grid
    p_stream[:, 7, 7] += 0.01 * p_stream[:, 7, 5]

    # Calculate streaming part and plot density for 4 timesteps
    for i in trange(4):
        streaming(p_stream)
        density = calculate_density(p_stream)
        v = calculate_velocity(p_stream)
        plot_density(density, i, False)
