import matplotlib.pyplot as plt
import GPUtil
import torch
import time
from pathlib import Path
import numpy as np

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ==============================================
# Constants (Optimized Memory Layout)
# ==============================================

# D2Q9 velocities [9 directions, xy coordinates]
E = torch.tensor([
    [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1]
], dtype=torch.float32, device=DEVICE)

# Weights for each direction
W = torch.tensor([
    4/9, 1/9, 1/9, 1/9, 1/9,
    1/36, 1/36, 1/36, 1/36
], dtype=torch.float32, device=DEVICE)

# Opposite directions
OPP = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], device=DEVICE)

# Reshaped for vectorized operations
E_opt = E.T.reshape(9, 2, 1, 1)  # [9,2,1,1] for broadcasting
W_opt = W.reshape(9, 1, 1)        # [9,1,1]

# Pre-computed streaming shifts
SHIFTS = torch.tensor([[int(e[0]), int(e[1])] for e in E], device=DEVICE)

# ==============================================
# Simulation Parameters
# ==============================================
NX, NY = 10000, 10000  # Grid size
NSTEPS = 10000
OMEGA = 1.0
TAU = 1 / OMEGA
LID_VELOCITY = 0.1
PLOT_DIR = Path("plots/milestones/m5")
PLOT_DIR.mkdir(exist_ok=True)

# ==============================================
# Core Functions (Fully Vectorized)
# ==============================================

def initialize():
    """Initialize with [9,NX,NY] layout"""
    rho = torch.ones((NX, NY), device=DEVICE) 
    u = torch.zeros((NX, NY, 2), device=DEVICE)
    f = equilibrium(rho, u)
    return f  # Returns [9,NX,NY]

def equilibrium(rho, u):
    """Vectorized equilibrium calculation"""
    # u: [NX,NY,2] -> [1,NX,NY,2]
    # E_opt: [9,2,1,1]
    cu = torch.einsum('dc,xyc->dxy', E, u)  # [9,NX,NY] 
    usqr = (u**2).sum(-1)  # [NX,NY]
    return rho * W_opt * (1 + 3*cu + 4.5*cu**2 - 1.5*usqr)  # [9,NX,NY]

def stream(f):
    """Single-kernel vectorized streaming"""
    return torch.stack([
        torch.roll(f[i], shifts=tuple(SHIFTS[i]), dims=(0,1))
        for i in range(9)
    ])

def collide_and_boundary(f):
    """Fused collision + boundary conditions"""
    # 1. Compute macroscopic
    rho = f.sum(dim=0)  # [NX,NY]
    u = torch.einsum('dc,dxy->xyc', E, f) / rho.unsqueeze(-1)
    
    # 2. Apply boundary conditions
    u[:,-1,0] = LID_VELOCITY  # Moving lid
    u[:,-1,1] = 0
    
    # 3. Bounce-back walls
    f[[2,5,6],:,0] = f[[4,7,8],:,0]  # Bottom
    f[[1,5,8],0,:] = f[[3,6,7],0,:]  # Left
    f[[3,6,7],-1,:] = f[[1,5,8],-1,:]  # Right
    
    # 4. Collision (in-place)
    feq = equilibrium(rho, u)
    f[:] = f - (1/TAU) * (f - feq)

# ==============================================
# Main Simulation
# ==============================================

def run_simulation():
    print(" Running simulation(Milestone 5 updated) with parameters :")
    print(f"NX: {NX}, NY: {NY}, NSTEPS: {NSTEPS}, OMEGA: {OMEGA}, TAU: {TAU}, LID_VELOCITY: {LID_VELOCITY}")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    f = initialize()
    start = time.time()
    
    for step in range(NSTEPS):
        f = stream(f)
        collide_and_boundary(f)
        
        if step % 1000 == 0:
            torch.cuda.synchronize()
            blups = (step * NX * NY) / (time.time() - start) / 1e9
            print(f"Step {step:5d}, BLUPS: {blups:.2f}")
    
    torch.cuda.synchronize()
    final_blups = (NSTEPS * NX * NY) / (time.time() - start) / 1e9
    mlups = (NSTEPS * NX * NY) / (time.time() - start) / 1e6
    print(f"Final BLUPS: {final_blups:.2f}")
    print(f"Final MLUPS: {mlups:.2f}")


# ==============================================
# Benchmarking and Plotting Function
# ==============================================
def benchmark_and_plot():
    grid_sizes = [1000, 3000, 5000, 10000, 15000, 20000, 25000, 30000]
    mlups_results = []
    power_draws = []
    speedups = []
    gpu_elapsed_times = []

    # CPU baseline (only for smallest grid size)
    cpu_device = torch.device("cpu")
    cpu_grid = 1000
    cpu_steps = 200
    f_cpu = torch.ones((9, cpu_grid, cpu_grid), device=cpu_device) * 0.1

    def stream_cpu(f):
        return torch.stack([
            torch.roll(f[i], shifts=(int(E[i, 0]), int(E[i, 1])), dims=(0, 1))
            for i in range(9)
        ])

    def collide_and_boundary_cpu(f):
        rho = f.sum(dim=0)
        u = torch.einsum('dc,dxy->xyc', E.to(cpu_device), f) / rho.unsqueeze(-1)
        u[:, -1, 0] = LID_VELOCITY
        u[:, -1, 1] = 0
        f[[2, 5, 6], :, 0] = f[[4, 7, 8], :, 0]
        f[[1, 5, 8], 0, :] = f[[3, 6, 7], 0, :]
        f[[3, 6, 7], -1, :] = f[[1, 5, 8], -1, :]
        feq = equilibrium(rho, u)
        f[:] = f - (1 / TAU) * (f - feq)

    start_cpu = time.time()
    for _ in range(cpu_steps):
        f_cpu = stream_cpu(f_cpu)
        collide_and_boundary_cpu(f_cpu)
    cpu_time = time.time() - start_cpu

    print("\n=== CPU BASELINE ({}x{}, {} steps) ===".format(cpu_grid, cpu_grid, cpu_steps))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"CPU baseline wall time: {cpu_time:.3f} s")

    for nx in grid_sizes:
        ny = nx
        steps = 2000
        f = torch.ones((9, nx, ny), device=DEVICE) * 0.1

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(steps):
            f = stream(f)
            collide_and_boundary(f)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        gpu_elapsed_times.append(elapsed)

        mlups = (steps * nx * ny) / elapsed / 1e6
        mlups_results.append(mlups)

        # Estimate power usage using GPUtil
        try:
            gpus = GPUtil.getGPUs()
            power_draw = gpus[0].powerDraw if gpus else 0
        except:
            power_draw = 0
        power_draws.append(power_draw)

        speedup = (cpu_grid**2 * cpu_steps / cpu_time) / (nx**2 * steps / elapsed)
        speedups.append(speedup)

        print(f"Grid: {nx}x{ny}, MLUPS: {mlups:.2f}, Power (W): {power_draw}, Speedup: {speedup:.2f}")

    baseline_gpu_mlups = mlups_results[0] if mlups_results else 1.0
    gpu_rel_speedups = [m / baseline_gpu_mlups for m in mlups_results]

    cpu_updates_baseline = cpu_grid * cpu_grid * cpu_steps
    cpu_est_times = []
    for nx in grid_sizes:
        ny = nx
        steps = 2000
        updates = nx * ny * steps
        scale = updates / cpu_updates_baseline
        cpu_est_times.append(cpu_time * scale)
    gpu_times = gpu_elapsed_times

    # Plot Grid Size vs MLUPS
    plt.figure()
    plt.plot(grid_sizes, mlups_results, marker='o', label='MLUPS')
    for gx, ml in zip(grid_sizes, mlups_results):
        plt.annotate(f"{ml:.0f}", (gx, ml), textcoords="offset points", xytext=(0,5), ha="center", fontsize=8)
    plt.title("Grid Size vs MLUPS (GPU Performance)")
    plt.xlabel("Grid Size (NX=NY)")
    plt.ylabel("MLUPS")
    plt.grid(True)
    plt.legend()
    plt.savefig(PLOT_DIR / "mlups_vs_grid.png")
    plt.show()

    # Optional: plot MLUPS per Watt
    mlups_per_watt = [mlups/p if p > 0 else 0 for mlups, p in zip(mlups_results, power_draws)]
    plt.figure()
    plt.plot(grid_sizes, mlups_per_watt, marker='x', color='green', label='MLUPS/Watt')
    for gx, eff in zip(grid_sizes, mlups_per_watt):
        plt.annotate(f"{eff:.2f}", (gx, eff), textcoords="offset points", xytext=(0,5), ha="center", fontsize=8)
    plt.title("Efficiency: MLUPS per GPU Watt")
    plt.xlabel("Grid Size (NX=NY)")
    plt.ylabel("MLUPS/Watt")
    plt.grid(True)
    plt.legend()
    plt.savefig(PLOT_DIR / "mlups_per_watt.png")
    plt.show()

    # Plot Speedup vs Grid Size
    plt.figure()
    plt.plot(grid_sizes, speedups, marker='s', color='purple', label='Speedup vs CPU')
    for gx, sp in zip(grid_sizes, speedups):
        plt.annotate(f"{sp:.1f}x", (gx, sp), textcoords="offset points", xytext=(0,5), ha="center", fontsize=8)
    plt.title("Speedup vs CPU (Baseline: 1000x1000 on CPU)")
    plt.xlabel("Grid Size (NX=NY)")
    plt.ylabel("Speedup")
    plt.grid(True)
    plt.legend()
    plt.savefig(PLOT_DIR / "speedup_vs_cpu.png")
    plt.show()

    # Plot relative GPU speedup vs grid size
    plt.figure()
    plt.plot(grid_sizes, gpu_rel_speedups, marker='d', color='orange', label='Relative GPU Speedup')
    plt.title("Relative GPU Speedup vs Grid Size (Normalized to Smallest Grid)", fontsize=12)
    plt.xlabel("Grid Size (NX=NY)")
    plt.ylabel("Speedup (MLUPS / MLUPS_baseline)")
    plt.grid(True)
    plt.legend()
    plt.savefig(PLOT_DIR / "gpu_rel_speedup_vs_grid.png")
    plt.show()

    # Annotated MLUPS vs Grid Size with values
    plt.figure()
    plt.plot(grid_sizes, mlups_results, marker='o', label='MLUPS')
    for i, mlups in enumerate(mlups_results):
        plt.annotate(f"{mlups:.1f}", (grid_sizes[i], mlups), textcoords="offset points", xytext=(0, 8), ha='center')
    plt.title("Grid Size vs MLUPS (Annotated)")
    plt.xlabel("Grid Size (NX=NY)")
    plt.ylabel("MLUPS")
    plt.grid(True)
    plt.legend()
    plt.savefig(PLOT_DIR / "annotated_mlups_vs_grid.png")
    plt.show()

    # Annotated MLUPS per Watt plot
    plt.figure()
    plt.plot(grid_sizes, mlups_per_watt, marker='x', color='green', label='MLUPS/Watt')
    for i, val in enumerate(mlups_per_watt):
        plt.annotate(f"{val:.2f}", (grid_sizes[i], val), textcoords="offset points", xytext=(0, 8), ha='center')
    plt.title("Efficiency: MLUPS per GPU Watt (Annotated)")
    plt.xlabel("Grid Size (NX=NY)")
    plt.ylabel("MLUPS/Watt")
    plt.grid(True)
    plt.legend()
    plt.savefig(PLOT_DIR / "annotated_efficiency.png")
    plt.show()

    # CPU vs GPU runtime breakdown bar chart
    x_idx = np.arange(len(grid_sizes))
    width = 0.35
    plt.figure(figsize=(10,5))
    plt.bar(x_idx - width/2, cpu_est_times, width, label='CPU-est (s)', color='gray')
    plt.bar(x_idx + width/2, gpu_times, width, label='GPU-measured (s)', color='teal')
    plt.xticks(x_idx, grid_sizes, rotation=45)
    plt.ylabel("Wall Time (s)")
    plt.xlabel("Grid Size (NX=NY)")
    plt.title("CPU vs GPU Runtime Breakdown (CPU scaled from baseline)", fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for xi, ct, gt in zip(x_idx, cpu_est_times, gpu_times):
        plt.annotate(f"{ct:.0f}", (xi - width/2, ct), textcoords="offset points", xytext=(0,3), ha="center", fontsize=7)
        plt.annotate(f"{gt:.1f}", (xi + width/2, gt), textcoords="offset points", xytext=(0,3), ha="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "cpu_gpu_runtime_breakdown.png")
    plt.show()

    return

if __name__ == "__main__":
    run_simulation()
    benchmark_and_plot()

