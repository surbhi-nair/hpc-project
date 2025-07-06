# m3_tests_runner.py

from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import json

DIRECTIONS = torch.tensor([
    [ 0,  0], [ 1,  0], [ 0,  1], [-1,  0], [ 0, -1],
    [ 1,  1], [-1,  1], [-1, -1], [ 1, -1]
], dtype=torch.float32)

WEIGHTS = torch.tensor([
    4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36
], dtype=torch.float32)

class LBMTest:
    def __init__(self, nx=15, ny=10, tau=0.6, test_case='test1'):
        self.nx = nx
        self.ny = ny
        self.ndir = 9
        self.tau = tau
        self.test_case = test_case
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.e = DIRECTIONS.to(self.device)
        self.w = WEIGHTS.to(self.device)
        self.f = torch.zeros((nx, ny, self.ndir), dtype=torch.float32, device=self.device)

        self.init_distribution()

    def init_distribution(self):
        rho = torch.ones((self.nx, self.ny), device=self.device) * 0.5
        u = torch.zeros((self.nx, self.ny, 2), device=self.device)
        if self.test_case == 'test1':
            rho[self.nx // 2, self.ny // 2] = 0.9
        elif self.test_case == 'test2':
            u[self.nx // 2, self.ny // 2, 0] = 0.05
        self.f = self.compute_equilibrium(rho, u)

    def compute_equilibrium(self, rho, u):
        eu = torch.einsum('xyi,ji->xyj', u, self.e)
        u2 = (u ** 2).sum(dim=2, keepdim=True)
        feq = rho.unsqueeze(-1) * self.w * (1 + 3 * eu + 4.5 * eu**2 - 1.5 * u2)
        return feq

    def compute_density(self):
        return self.f.sum(dim=2)

    def compute_velocity(self):
        rho = self.compute_density().unsqueeze(-1)
        momentum = (self.f.unsqueeze(-1) * self.e).sum(dim=2)
        u = torch.zeros_like(momentum)
        mask = rho.squeeze(-1) > 0
        u[mask] = momentum[mask] / rho[mask]
        return u

    def collide(self):
        rho = self.compute_density()
        u = self.compute_velocity()
        feq = self.compute_equilibrium(rho, u)
        self.f += -(1.0 / self.tau) * (self.f - feq)

    def stream(self):
        f_new = torch.empty_like(self.f)
        for i in range(self.ndir):
            dx, dy = self.e[i].int()
            f_new[..., i] = torch.roll(self.f[..., i], shifts=(dx.item(), dy.item()), dims=(0, 1))
        self.f = f_new

    def save_density_field(self, step, out_dir):
        rho = self.compute_density().cpu().numpy()
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.imshow(rho.T, origin="lower", cmap="viridis")
        plt.colorbar(label="Density")
        plt.xlabel("Lattice X Position (grid index)")
        plt.ylabel("Lattice Y Position (grid index)")
        plt.title(f"Density at Step {step}")
        plt.savefig(out_dir / f"density_step_{step:04d}.png")
        plt.close()

    def save_velocity_field(self, step, out_dir):
        u = self.compute_velocity().cpu().numpy()
        X, Y = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing='ij')
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.quiver(X, Y, u[..., 0], u[..., 1], scale=0.005, scale_units='xy')
        plt.xlabel("Lattice X Position (grid index)")
        plt.ylabel("Lattice Y Position (grid index)")
        plt.title(f"Velocity Field at Step {step}")
        plt.savefig(out_dir / f"velocity_step_{step:04d}.png")
        plt.close()

    def run(self, steps=50, save_every=5):
        base_dir = Path("plots") / f"m3_{self.test_case}"
        velocities = []
        for step in range(steps):
            self.collide()
            self.stream()
            # record mean velocity magnitude
            mean_u = self.compute_velocity().norm(dim=-1).mean().item()
            velocities.append(mean_u)
            if step % save_every == 0:
                self.save_density_field(step, base_dir)
                self.save_velocity_field(step, base_dir)
        
        # save velocities to JSON for later plotting
        base_dir.mkdir(parents=True, exist_ok=True)
        with open(base_dir/"velocity_magnitudes.json","w") as f:
            json.dump(velocities,f)

if __name__ == "__main__":
    for test in ['test1', 'test2']:
        sim = LBMTest(test_case=test)
        sim.run()