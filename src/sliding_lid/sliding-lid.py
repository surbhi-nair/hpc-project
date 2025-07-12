import os
import torch
import argparse
import lbm
import matplotlib.pyplot as plt
import time

from tqdm import trange
from typing import Optional
from utils import theoretical_viscosity, save_streamplot

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PLOT_FLAG = False  # Set to True to enable plotting
PLOT_DIR = ""
if PLOT_FLAG:
    PLOT_DIR = "plots/sliding_lid"
    os.makedirs(PLOT_DIR, exist_ok=True)

def sliding_lid(
        proba_density,
        lid_velocity,
        omega: Optional[float] = 0.8,
        steps: Optional[int] = 10000,
        gif: Optional[bool] = False
) -> None:
    """
    Perform the sliding lid experiment. Given a closed box filled with liquid, simulate
        the scenario in which the lid of the box moves with a constant velocity to the
        right. We assume initial conditions of density ρ = 1, and both x- and y-components
        of velocity equal to zero.

    Args:
        proba_density (torch.Tensor): Probability density function of Lattice.
        lid_velocity: (torch.Tensor): The velocity vector of the moving boundary.
                Currently, a 2D-ndarray containing one value for the x-component
                and one value for the y-component of the velocity.
        omega (Optional[float]): Collision frequency.
        steps (Optional[int]): Simulation steps.
        gif (Optional[bool], default=False): If True, we keep intermediate plots
            to generate gif of the flow

    Returns:
        None.
    """

    # Get the dimensions
    _, x_shape, y_shape = proba_density.shape

    # Choose initial values for density ρ(0) = 1.0 at time t=0
    density = torch.ones((x_shape, y_shape), dtype=torch.float32, device=DEVICE)
    # Choose initial value for velocity u(0) = 0.0 at time t=0
    velocity = torch.zeros((2, x_shape, y_shape), dtype=torch.float32, device=DEVICE)
    # Initialize probability density based on equilibrium
    proba_density = lbm.calculate_equilibrium_distro(density, velocity)

    # Initialize dictionary to hold x-velocity values at different timesteps
    # and keep the initial velocity
    vx_dict = {0: torch.moveaxis(velocity[0], 0, 1)[:, x_shape // 2]}
    # Keep velocity every 500 steps if not generating gif, else every 100
    keep_every_steps = 500 if not gif else 100

    # May be needed to generate gif
    if gif and PLOT_FLAG:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))

    start_time = time.time()

    # For each iteration step
    for step in trange(steps, desc="Sliding lid"):
        # Calculate density
        density = lbm.calculate_density(proba_density)
        # Calculate velocity
        velocity = lbm.calculate_velocity(proba_density)
        # Perform collision/relaxation
        lbm.collision_relaxation(proba_density, velocity, density, omega=omega)
        # Keep the probability density function pre-streaming
        pre_stream_proba_density = proba_density.clone()
        # Streaming
        lbm.streaming(proba_density)
        # Apply boundary conditions on the bottom rigid wall
        lbm.rigid_wall(proba_density, pre_stream_proba_density, "lower")
        # Apply boundary condition on the top moving wall
        lbm.moving_wall(proba_density, pre_stream_proba_density,
                        lid_velocity, density, "upper")
        # Apply boundary conditions on the left rigid wall
        lbm.rigid_wall(proba_density, pre_stream_proba_density, "left")
        # Apply boundary conditions on the right rigid wall
        lbm.rigid_wall(proba_density, pre_stream_proba_density, "right")

        if step % keep_every_steps == 0:
            # Keep the velocity in a slice on the axis that is perpendicular to the moving
            # boundary, the shape of vx_dict[<step>] is (lattice_size,)
            vx_dict[step] = torch.moveaxis(velocity[0], 0, 1)[:, x_shape // 2]

            if gif and PLOT_FLAG:
                save_streamplot(velocity, step, ax)

    total_time = time.time() - start_time
    total_lattice_updates = steps * x_shape * y_shape
    blups = total_lattice_updates / total_time / 1e9
    print(f"\nBLUPS Performance: {blups:.3f} Billion Lattice Updates Per Second")

    if PLOT_FLAG:
        # Plotting
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))

        # Figure title
        fig.suptitle("Sliding Lid", fontsize=16)

        # Velocity field streamplot
        lbm.plot_velocity_field(velocity, fig, ax)

        plt.legend()

        # Save plot
        print("Saving plot...")
        plt.savefig(PLOT_DIR + f"sliding_lid_velocity_field_{int(omega * 10)}_{steps}_{x_shape}_{y_shape}")

        # # Display plot
        # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Sliding lid experiment",
        description="Run Sliding lid experiment."
    )

    parser.add_argument(
        "-o",
        "--omega",
        help="The collision frequency. Default value is 0.8.",
        type=float,
        default=0.8
    )

    parser.add_argument(
        '-v',
        "--velocity",
        help="The velocity of the moving lid.",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "-s",
        "--steps",
        help="The simulation steps. Default value is 10000.",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "-i",
        "--gif",
        help="Generate gif.",
        default=False,
        action="store_true"
    )

    parser.add_argument('-g',
                        '--grid_size',
                        nargs='+',
                        help='Space separated list of grid size (dim_0, dim_1). For '
                             'example: -g 50 50',
                        type=int,
                        default=(300, 300)
                        )

    args = parser.parse_args()

    # Grid dimensions
    dim_0, dim_1 = args.grid_size
    # Define lid velocity
    w = torch.tensor([args.velocity, 0.], dtype=torch.float32, device=DEVICE)
    # Initialize the grid
    p = torch.zeros((9, dim_0, dim_1), dtype=torch.float32, device=DEVICE)

    viscosity = theoretical_viscosity(args.omega)
    reynolds = dim_0 * args.velocity / viscosity

    print("Running Sliding Lid simulation with the following setup:"
          f"\n\tDEVICE: \t{DEVICE}"
          f"\n\tGrid size: \t\t{dim_0} x {dim_1}"
          f"\n\tCollision Frequency: \t{args.omega}"
          f"\n\tViscosity: \t\t{viscosity:.4f}"
          f"\n\tLid velocity: \t\t{args.velocity}"
          f"\n\tReynolds number: \t{reynolds:.2f}"
          f"\n\tSimulation steps: \t{args.steps}"
          )

    # Run simulation
    sliding_lid(p, w,
                omega=args.omega,
                steps=args.steps,
                gif=args.gif)

    print("\nSimulation completed.")
