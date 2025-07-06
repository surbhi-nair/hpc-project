import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# D2Q9 direction vectors
DIRECTIONS = torch.tensor([
    [ 0,  0],  # 0: rest
    [ 1,  0],  # 1: east
    [ 0,  1],  # 2: north
    [-1,  0],  # 3: west
    [ 0, -1],  # 4: south
    [ 1,  1],  # 5: northeast
    [-1,  1],  # 6: northwest
    [-1, -1],  # 7: southwest
    [ 1, -1],  # 8: southeast
], dtype=torch.int32)

'''
Milestone 2: Streaming Operator
In this milestone, we implement the streaming step of the Lattice Boltzmann Method (LBM) for a 2D grid.
The streaming step shifts the distribution functions in their respective directions, simulating particle movement.
We only focus on the streaming, without collision or boundary conditions so the particles only move - no interactions.
The grid is initialized with a single directional injection of particles moving to the right at the center.
The simulation runs for a specified number of steps, saving the velocity and density fields at regular intervals
'''

class LBMGrid2D:
    def __init__(self, nx=15, ny=10, device=None):
        """
        Initialize the 2D Lattice Boltzmann grid.

        Args:
            nx (int): Number of lattice nodes in x-direction.
            ny (int): Number of lattice nodes in y-direction.
            device (torch.device, optional): Device to run simulation on (CPU or CUDA).
        """
        self.nx = nx
        self.ny = ny
        self.ndir = 9
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize distribution function f
        self.f = torch.zeros((nx, ny, self.ndir), dtype=torch.float32, device=self.device)
        self.e = DIRECTIONS.to(self.device)

        self.initialize_distribution()

    def initialize_distribution(self):
        """
        Initialize the distribution function f with a single directional injection.
        Injects unit mass in the east direction at the center of the grid.
        """
        # Start with particles moving to the right at center
        self.f[self.nx // 2, self.ny // 2, 1] = 1.0

    def compute_density(self):
        """
        Compute the fluid density ρ at each lattice node.

        Returns:
            torch.Tensor: 2D tensor of shape (nx, ny) representing fluid density.
        """
        # Sum over directions
        return self.f.sum(dim=2)

    def compute_velocity(self):
        """
        Compute the macroscopic velocity vector u at each lattice node.

        Returns:
            torch.Tensor: 3D tensor of shape (nx, ny, 2) containing velocity vectors.
        """
        rho = self.compute_density().unsqueeze(-1)  # shape: (nx, ny, 1)
        # Multiply each direction f_i by its velocity e_i, then sum over directions
        momentum = (self.f.unsqueeze(-1) * self.e).sum(dim=2)  # shape: (nx, ny, 2)

        u = torch.zeros_like(momentum)
        mask = rho.squeeze(-1) > 0
        u[mask] = momentum[mask] / rho[mask]
        return u

    def stream(self):
        """
        Perform the streaming step of the Lattice Boltzmann method.
        Shifts the distribution functions in their respective directions using periodic boundaries.
        """
        # Create a new tensor to hold the shifted distribution functions
        f_new = torch.empty_like(self.f)
        # Shift each direction's distribution function according to its velocity vector
        for i in range(self.ndir):
            # Get the shift amounts for the current direction
            dx, dy = self.e[i]
            # Use torch.roll to shift the distribution function in the specified direction
            # Note: dx and dy are integers, so we convert them to int
            f_new[..., i] = torch.roll(self.f[..., i], shifts=(int(dx.item()), int(dy.item())), dims=(0, 1))
        self.f = f_new

    def save_velocity_field(self, step=0, out_dir=None):
        """
        Save a quiver plot of the velocity field at the current step.

        Args:
            step (int): Current simulation step (used for filename).
            out_dir (Path or None): Directory to save plot. Defaults to project_root/plots.
        """
        if out_dir is None:
            # resolve "plots/" relative to the project root
            current_file = Path(__file__).resolve()
            project_root = current_file.parents[2]  # go up from milestone2.py → src → project root
            out_dir = project_root / "plots/m2"

        out_dir.mkdir(parents=True, exist_ok=True)

        u = self.compute_velocity().cpu().numpy()
        X, Y = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing='ij')

        plt.figure(figsize=(6, 4))
        plt.quiver(X, Y, u[..., 0], u[..., 1], scale=1, scale_units='xy')
        plt.xlabel("Lattice X Position (grid index)")
        plt.ylabel("Lattice Y Position (grid index)")
        plt.title(f"Velocity Field at Step {step}")
        plt.savefig(out_dir / f"m2_velocity_step_{step:04d}.png")
        plt.close()

    def save_density_field(self, step=0, out_dir=None):
        """
        Save a heatmap of the fluid density at the current step.

        Args:
            step (int): Current simulation step (used for filename).
            out_dir (Path or None): Directory to save plot. Defaults to project_root/plots.
        """
        if out_dir is None:
            current_file = Path(__file__).resolve()
            project_root = current_file.parents[2]
            out_dir = project_root / "plots/m2"

        out_dir.mkdir(parents=True, exist_ok=True)
        rho = self.compute_density().cpu().numpy()
        plt.imshow(rho.T, origin="lower", cmap="viridis")
        plt.colorbar(label="Density")
        plt.xlabel("Lattice X Position (grid index)")
        plt.ylabel("Lattice Y Position (grid index)")
        plt.title(f"Density at Step {step}")
        plt.savefig(out_dir / f"m2_density_step_{step:04d}.png")
        plt.close()

    def run(self, steps=20, save_every=5):
        """
        Run the LBM simulation for a specified number of steps.

        Args:
            steps (int): Total number of simulation steps to perform.
            save_every (int): Save output every n steps.
        """
        for step in range(steps):
            self.stream()
            if step % save_every == 0:
                self.save_velocity_field(step)
                self.save_density_field(step)

# Entry point
if __name__ == "__main__":
    sim = LBMGrid2D(nx=15, ny=10)
    sim.run(steps=20, save_every=5)