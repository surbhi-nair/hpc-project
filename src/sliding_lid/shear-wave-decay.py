import os
import argparse
import matplotlib.pyplot as plt
import lbm
import torch
from typing import List, Optional
from tqdm import trange
from constants import *
from utils import theoretical_decay_calcs, theoretical_viscosity

if True:
    PLOT_DIR = Path("plots/shear_wave_decay/")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def sin_density(
    probab_density_f,
    omega: Optional[float] = 0.5,
    steps: Optional[int] = 1000,
    initial_density: Optional[float] = 0.8,
) -> None:
    """
    Shear wave decay with initial distribution for velocity u(r,0)=1 and for
        density ρ(r,0) = ρ_0 + ε * sin(2π * x / Lx) (assuming symmetrical grid).
    Args:
        probab_density_f (np.ndarray): Probability density function of Lattice
            Boltzmann Equation.
        omega (Optional[float]): Collision frequency. Default value is 0.5
        initial_density (Optional[float] ): Initial mass density at each position of the grid.
            Default = 0.8.
        steps (Optional[int] ): Simulation steps. Default value is 2000
    Returns:
        None.
    """
    _, x_shape, y_shape = probab_density_f.shape
    x = np.arange(x_shape)
    x = torch.tensor(x, dtype=torch.float32, device=DEVICE)

    # Initialize density and velocity
    # Initialize velocity as zeros in the y-direction and as the sinusoidal wave in the x-direction
    velocity = torch.zeros((2, x_shape, y_shape), dtype=torch.float32, device=DEVICE)

    rho_y_1d = initial_density + SW_EPSILON * torch.sin(2 * np.pi / x_shape * x)
    density = rho_y_1d.repeat(y_shape, 1).T

    # Initialize proba density based on equilibrium
    probab_density_f = lbm.compute_equilibrium(density, velocity)

    # Choose a specific point in the grid and monitor each density over time
    point = (1, 0)

    # Initialize array to hold sinus wave amplitude at the monitored point
    sinus_amp = [density[point].item()]

    # Loop over a number of simulation steps
    for _ in trange(steps, desc="Density Simulation"):
        # Perform the standard streaming procedure
        lbm.streaming(probab_density_f)

        # Calculate density
        density = lbm.compute_density(probab_density_f)

        # Calculate velocity
        velocity = lbm.compute_velocity(probab_density_f)

        # Perform collision and update
        lbm.collision_relaxation(probab_density_f, velocity, density, omega=omega)

        # Measure the density at the monitored point
        sinus_amp.append(density[point].item())

    # Initialize a figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 18))

    # Figure title
    fig.suptitle(
        "Shear wave decay with sinusoidal density on y-axis of the " "lattice",
        fontsize=16,
    )
    _ = ax.set_title(
        "Sinusoidal wave decay at each individual point on the "
        "lattice for omega equal to {}".format(omega)
    )
    _ = ax.set_ylabel("Wave amplitude")
    _ = ax.set_xlabel("Time step")
    _ = ax.legend()
    _ = ax.grid()
    _ = ax.plot(sinus_amp, label="amp")

    print("Plots saved in", PLOT_DIR,"swd_sinusoidal_density.png")
    plt.savefig(PLOT_DIR / "swd_sinusoidal_density.png")
    plt.show()


def sin_velocity(
    probab_density_f,
    omega: Optional[float] = 1.0,
    steps: Optional[int] = 1000,
) -> None:
    """
    Shear wave decay with initial distribution for density ρ(r,0)=1 and for
        velocity ux(r,0)=ε*sin(2π*y/Ly). Periodic boundary conditions.
    Args:
        probab_density_f (nd.ndarray): Probability density function of Lattice
            Boltzmann Equation.
        omega (Optional[float]): Collision frequency.
        steps (Optional[int]): Simulation steps.
    Returns:
        None.
    """
    # Get the dimensions
    _, x_shape, y_shape = probab_density_f.shape
    # Create the y-axis, avoid meshgrid here
    y = torch.arange(y_shape, dtype=torch.float32, device=DEVICE)

    # Initialize density
    density = torch.ones((x_shape, y_shape), dtype=torch.float32, device=DEVICE)

    # Initialize velocity as zeros in the y-direction and as the sinusoidal
    # wave in the x-direction
    u_y = torch.zeros((x_shape, y_shape), dtype=torch.float32, device=DEVICE)
    # Create 1d array of sinusoidal wave
    u_x_1d = SW_EPSILON * torch.sin(2 * np.pi / y_shape * y)
    # Broadcast the array x_shape number of times
    u_x = u_x_1d.repeat(x_shape, 1)

    # Initialize array to hold sinus wave amplitude as it decays, e.g.
    # every 10 steps
    sinus_amp = []
    # Initialize dictionary to hold the velocity sinus wave as it decays,
    # e.g. every 10 steps
    sinus_waves = {}
    # Initialize array to hold the simulation steps. Will be used later
    # in plotting.
    sinus_steps = []

    # Reformat velocity so that it can be used later
    velocity = torch.stack((u_x, u_y), dim=0)

    # Initialize proba density based on equilibrium
    probab_density_f = lbm.compute_equilibrium(density, velocity)

    # Loop over the simulation steps
    for step in trange(steps, desc="Velocity Simulation"):

        # Perform the standard streaming procedure
        lbm.streaming(probab_density_f)

        # Calculate density
        density = lbm.compute_density(probab_density_f)

        # Calculate velocity
        velocity = lbm.compute_velocity(probab_density_f)

        # Perform collision and update
        lbm.collision_relaxation(probab_density_f, velocity, density, omega=omega)

        if step % 100 == 0:
            # Get the u_x velocity of one sinus wave
            sinus_steps.append(step)
            u_wave = velocity[0, 0, :].detach().cpu().numpy()
            sinus_waves[step] = u_wave
            sinus_amp.append(torch.max(velocity[0, 0, :]).item())

    # Calculate the theoretical decay based on viscosity
    theoretical_data = theoretical_decay_calcs(
        initial_amp=SW_EPSILON, omega=1, length=y_shape, timestep=sinus_steps
    )
    print("Theoretical viscosity: {}".format(theoretical_viscosity(omega=1)))

    # Calculate the viscosity based on experimental data with a Dt of 100 steps
    # The exponential decay curve is given by the equation
    # a(t) = a(0) * exp(-v (2*pi / L)^2 * t)
    # which means that the experimental viscosity can be calculated as
    # v = -(ln(a(t)/a(0)) / ( (2*pi / L)^2 * t)
    experimental_viscosity = -np.log(sinus_amp[1] / sinus_amp[0]) / (
        (2 * np.pi / y_shape) ** 2 * 100
    )
    print("Experimental viscosity: {}".format(experimental_viscosity))
    # Initialize a figure containing 2 axes objects

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 10))

    # Create a plot with the sinusoidal waves decay over time for a single
    # row in the lattice
    # Plot over steps of two to avoid cluttering
    for step in sinus_steps[::2]:
        _ = ax[0].plot(sinus_waves[step], label="Step: {}".format(step))

    # Figure title
    fig.suptitle("Shear wave decay with sinusoidal velocities on x-axis", fontsize=16)

    # Create a plot with the sinusoidal waves decaying over time for a single
    # lattice
    _ = ax[0].set_title(
        "Sinusoidal waves over time for a single row in the "
        "lattice \nover different simulation steps"
    )
    _ = ax[0].set_ylabel("Wave amplitude")
    _ = ax[0].set_xlabel("Lattice X Coordinate")
    _ = ax[0].legend()

    # Create a plot with the exponential decay of the amplitude of the
    # sinusoidal waves
    _ = ax[1].plot(sinus_amp, label="Experimental decay", linewidth=3)
    _ = ax[1].plot(
        theoretical_data, label="Theoretical decay", marker="o", linestyle="dashed"
    )
    _ = ax[1].set_xticks(range(len(sinus_steps)))
    _ = ax[1].set_xticklabels(sinus_steps, rotation=45)
    _ = ax[1].set_title("Shear wave decay")
    _ = ax[1].set_ylabel("Wave amplitude")
    _ = ax[1].set_xlabel("Simulation steps")
    _ = ax[1].legend()

    # Save plots
    print("Plots saved in", PLOT_DIR)
    plt.savefig(PLOT_DIR / "swd_sinusoidal_velocity.png")


def swd_test_omega(
    probab_density_f, omegas: List[float], steps: Optional[int] = 1000
) -> None:
    """
    Plot the exponential decay of a shear wave decay with initial
        distribution for density ρ(r,0)=1 and for velocity
        ux(r,0)=ε*sin(2π*y/Ly) for different omega values.
        Periodic boundary conditions.
    Args:
        probab_density_f (np.ndarray): Probability density function of Lattice
            Boltzmann Equation.
        omegas (Optional[float]): The values of collision frequency for which the experiment will run.
        steps (Optional[int]): Simulation steps.
    Returns:
        None.
    """
    # Get the dimensions
    _, x_shape, y_shape = probab_density_f.shape
    # Create the y-axis, avoid meshgrid here
    y = torch.arange(y_shape, dtype=torch.float32, device=DEVICE)

    # Initialize density
    density = torch.ones((x_shape, y_shape), dtype=torch.float32, device=DEVICE)

    # Initialize velocity as zeros in the y-direction and as the sinusoidal
    # wave in the x-direction
    u_y = torch.zeros((x_shape, y_shape), dtype=torch.float32, device=DEVICE)
    # Create 1d array of sinusoidal wave
    u_x_1d = SW_EPSILON * torch.sin(2 * np.pi / y_shape * y)
    # Broadcast the array x_shape number of times
    u_x = u_x_1d.repeat(x_shape, 1)

    # Initialize dictionary to hold the theoretical viscosity
    theoretical_visc = {}
    # Initialize dictionary to hold the experimental viscosity
    experimental_visc = {}
    # Initialize dictionary to hold sinus wave amplitude as it decays
    sinus_amp = {}

    for omega in omegas:
        # Initialize array to hold sinus wave amplitude for the iterated omega
        sinus_amp[omega] = []

        theoretical_visc[omega] = theoretical_viscosity(omega)

        # Reformat velocity so that it can be used later
        velocity = torch.stack((u_x, u_y), dim=0)

        # Initialize proba density based on equilibrium
        probab_density_f = lbm.compute_equilibrium(density, velocity)
        # Loop over a number of simulation steps

        for step in trange(steps, desc="Omega: {}".format(omega)):

            # Perform the standard streaming procedure
            lbm.streaming(probab_density_f)

            # Calculate density
            density = lbm.compute_density(probab_density_f)

            # Calculate velocity
            velocity = lbm.compute_velocity(probab_density_f)

            # Perform collision and update
            lbm.collision_relaxation(probab_density_f, velocity, density, omega=omega)

            if step % 10 == 0:
                # Get the u_x velocity of one sinus wave
                u_wave = velocity[0, 0, :]
                sinus_amp[omega].append(torch.max(u_wave).item())

        experimental_visc[omega] = -np.log(
            sinus_amp[omega][1] / sinus_amp[omega][0]
        ) / ((2 * np.pi / y_shape) ** 2 * 10)

    # Initialize figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 18))
    _ = ax.set_ylabel("Wave amplitude")
    _ = ax.set_xlabel("Time step / 10")
    for omega in omegas:
        _ = ax.plot(
            sinus_amp[omega],
            label="Experimental decay " "for ω = {}".format(omega),
            linewidth=3,
        )

    _ = ax.legend()

    _ = ax.set_title("Shear wave decay for different ω")
    _ = ax.grid()
    # Save plots
    print("Plots saved in", PLOT_DIR,"swd_sinusoidal_velocity_var_omegas.png")
    plt.savefig(PLOT_DIR / "swd_sinusoidal_velocity_var_omegas.png")

    theoretical = []
    experimental = []
    for omega in omegas:
        theoretical.append(theoretical_visc[omega])
        experimental.append(experimental_visc[omega])
    # Print theoretical and experimental viscosities for each omega in tabular format
    print("\nTheoretical and Experimental Viscosities for each Omega:")
    print(f"{'Omega':<10} {'Theoretical Viscosity':<25} {'Experimental Viscosity':<25}")
    print("-" * 60)
    for omega in omegas:
        print(f"{omega:<10} {theoretical_visc[omega]:<25.4f} {experimental_visc[omega]:<25.4f}")
    # New plotting block with annotation labels for each omega
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
    ax.set_title("Theoretical vs Experimental Viscosity vs Omega ω", fontsize=14)
    ax.plot(omegas, theoretical, label="Theoretical Viscosity", linestyle="--", color='black')
    ax.scatter(omegas, experimental, label="Experimental Viscosity", color='red')

    for i, omega in enumerate(omegas):
        ax.annotate(f"{omega:.2f}", (omegas[i], experimental[i]), textcoords="offset points",
                    xytext=(0, 5), ha='center', fontsize=9, color='blue')

    ax.set_xlabel("Omega ω", fontsize=12)
    ax.set_ylabel("Viscosity ν", fontsize=12)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    # Save plots
    print("Viscosity plots saved in", PLOT_DIR)
    plt.savefig(PLOT_DIR / "swd_sinusoidal_velocity_var_omegas_viscosity.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Shear Wave Decay", description="Run Shear Wave Decay experiment"
    )
    parser.add_argument(
        "-d",
        "--density",
        help="Execute sinusoidal density experiment.",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "-v",
        "--velocity",
        help="Execute sinusoidal velocity experiment.",
        default=True,
        action="store_true",
    )

    parser.add_argument(
        "-f",
        "--frequency",
        help="Execute multiple sinusoidal velocity experiments with varying"
        "collision frequency values.",
        default=True,
        action="store_true",
    )

    parser.add_argument(
        "-fv",
        "--frequency_values",
        nargs="+",
        help="Space separated list of omega values. For " "example: -fv 1.1 1.2 1.4",
        type=float,
        default="-1",
    )

    parser.add_argument(
        "-g",
        "--grid_size",
        nargs="+",
        help="Space separated list of grid size (dim_0, dim_1). For "
        "example: -g 50 50",
        type=int,
        default=(600, 600),
    )

    parser.add_argument(
        "-o",
        "--omega",
        help="The collision frequency. Default value is 1.",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "-s",
        "--steps",
        help="The simulation steps. Default value is 2000.",
        type=int,
        default=2000,
    )

    # Parse arguments
    args = parser.parse_args()

    # Grid dimensions
    dim_0, dim_1 = args.grid_size
    # Initialize the grid
    p = torch.zeros((9, dim_0, dim_1), dtype=torch.float32, device=DEVICE)
    print("Grid size: {} x {}".format(dim_0, dim_1))

    # If none is provided, run for velocity
    if not (args.velocity or args.density or args.frequency):
        args.velocity = True

    if args.velocity:
        # Run the shear wave decay experiment with sinusoidal velocity
        print(
            "Running shear wave decay simulation with sinusoidal velocity with omega {}:".format(
                args.omega
            )
        )
        sin_velocity(p, args.omega, args.steps)

    elif args.density:
        print(
            "Running shear wave decay simulation with sinusoidal density with omega {}:".format(
                args.omega
            )
        )
        # Run the shear wave decay experiment with sinusoidal density
        sin_density(p, args.omega, args.steps)

    if args.frequency:
        print("Running shear wave decay simulation for various omegas:")
        omegas = args.frequency_values
        if omegas == -1: # default
            omegas = [0.6, 0.8, 1.0, 1.2, 1.4, 1.8]
        swd_test_omega(p, omegas, args.steps)
