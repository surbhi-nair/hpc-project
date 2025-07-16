import os
import torch
import argparse
import lbm
import matplotlib.pyplot as plt
import time

from tqdm import trange
from typing import Optional
from utils import theoretical_viscosity, save_streamplot
from constants import *
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.checkpoint import checkpoint

torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster matrix multiplications
torch.set_float32_matmul_precision('high')  # Enable TF32 for better performance

if PLOT_FLAG:
    PLOT_DIR = Path("plots/sliding_lid/")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

@torch.compile(mode="reduce-overhead")
def sliding_lid(
        probab_density_f,
        lid_velocity,
        omega: Optional[float] = 0.8,
        steps: Optional[int] = 10000
) -> None:
    """
    Perform the sliding lid experiment. Given a closed box filled with liquid, simulate
        the scenario in which the lid of the box moves with a constant velocity to the
        right. We assume initial conditions of density ρ = 1, and both x- and y-components
        of velocity equal to zero.

    Args:
        probab_density_f (torch.Tensor): Probability density function of Lattice.
        lid_velocity: (torch.Tensor): The velocity vector of the moving boundary.
                Currently, a 2D-ndarray containing one value for the x-component
                and one value for the y-component of the velocity.
        omega (Optional[float]): Collision frequency.
        steps (Optional[int]): Simulation steps.

    Returns:
        None.
    """
    _, x_shape, y_shape = probab_density_f.shape
    rho = torch.empty((x_shape, y_shape), dtype=torch.float32, device=DEVICE)
    velocity = torch.empty((2, x_shape, y_shape), dtype=torch.float32, device=DEVICE)
    probab_density_f = lbm.compute_equilibrium(rho, velocity)

    # Initialize dictionary to hold x-velocity values at different timesteps
    # and keep the initial velocity
    vx_dict = {0: torch.moveaxis(velocity[0], 0, 1)[:, x_shape // 2]}
    # Keep velocity every 500 steps if not generating gifs, else every 100
    keep_every_steps = 500 if not PLOT_FLAG else 100

    # to generate plots 
    if PLOT_FLAG:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #              record_shapes=True,
    #              profile_memory=True,
    #              with_stack=True) as prof:
    for step in range(steps):
        pre_stream_probab_density_f = torch.empty_like(probab_density_f)
        pre_stream_probab_density_f.copy_(probab_density_f)

        with torch.no_grad():
            # Calculate density
            lbm.compute_density(probab_density_f, out=rho)
            # Calculate velocity
            lbm.compute_velocity(probab_density_f, out=velocity)

        # Use checkpointing for collision_relaxation for memory efficiency
        # def collision_fn(prob_density, vel, dens, omega_val):
        #     lbm.collision_relaxation(prob_density, vel, dens, omega=omega_val)
        #     return prob_density

        # probab_density_f = checkpoint(collision_fn, probab_density_f, velocity, rho, omega)
        lbm.collision_relaxation(probab_density_f, velocity, rho, omega=omega)

        # Streaming
        lbm.streaming(probab_density_f)
        # Apply boundary conditions on the bottom rigid wall
        lbm.rigid_wall(probab_density_f, pre_stream_probab_density_f, "lower")
        # Apply boundary condition on the top moving wall
        lbm.moving_wall(probab_density_f, pre_stream_probab_density_f,
                        lid_velocity, rho, "upper")
        # Apply boundary conditions on the left rigid wall
        lbm.rigid_wall(probab_density_f, pre_stream_probab_density_f, "left")
        # Apply boundary conditions on the right rigid wall
        lbm.rigid_wall(probab_density_f, pre_stream_probab_density_f, "right")

        if step % keep_every_steps == 0 and PLOT_FLAG:
            # Keep the velocity in a slice on the axis that is perpendicular to the moving
            # boundary, the shape of vx_dict[<step>] is (lattice_size,)
            vx_dict[step] = torch.moveaxis(velocity[0], 0, 1)[:, x_shape // 2]
            save_streamplot(velocity, step, ax)

    # print("\n==== Torch Profiler Summary ====")
    # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

    end_event.record()
    torch.cuda.synchronize()  # Wait for GPU to finish
    gpu_time_sec = start_event.elapsed_time(end_event) / 1000  # ms -> s
    total_updates = steps * x_shape * y_shape
    blups = total_updates / gpu_time_sec / 1e9
    
    print(f"========= Optimized GPU Performance: {blups:.3f} BLUPS (GPU Time: {gpu_time_sec:.3f} s) =========")

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
        "--PLOT_FLAG",
        help="Generate PLOT_FLAG.",
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
                steps=args.steps)

    print("\nSimulation completed.")
