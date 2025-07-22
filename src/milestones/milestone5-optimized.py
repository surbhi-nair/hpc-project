import matplotlib.pyplot as plt
import GPUtil
import torch
import time
from pathlib import Path
import numpy as np
import argparse
from datetime import datetime

# Enable performance optimizations
torch.set_float32_matmul_precision('high')  # Enable TF32 for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ==============================================
# Constants (Optimized Memory Layout)
# ==============================================

# D2Q9 velocities [9 directions, xy coordinates]
E = torch.tensor(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=torch.float32,
    device=DEVICE,
)

# Weights for each direction
W = torch.tensor(
    [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
    dtype=torch.float32,
    device=DEVICE,
)

# Opposite directions
OPP = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], device=DEVICE)

# Reshaped for vectorized operations
E_opt = E.T.reshape(9, 2, 1, 1)  # [9,2,1,1] for broadcasting
W_opt = W.reshape(9, 1, 1)  # [9,1,1]

# Pre-computed streaming shifts
SHIFTS = torch.tensor([[int(e[0]), int(e[1])] for e in E], device=DEVICE)

# ==============================================
# Simulation Parameters
# ==============================================
NX, NY = 5000, 5000  # Optimized grid size for better performance measurement
NSTEPS = 2000
OMEGA = 1.0
TAU = 1 / OMEGA
LID_VELOCITY = 0.1
PLOT_DIR = Path("plots/milestones/m5")
PLOT_DIR.mkdir(exist_ok=True, parents=True)

PLOT_DIR_B = Path("plots/milestones/m5_benchmark")
PLOT_DIR_B.mkdir(exist_ok=True, parents=True)


# ==============================================
# Core Functions (Fully Vectorized)
# ==============================================


def initialize():
    """Initialize with [9,NX,NY] layout"""
    rho = torch.ones((NX, NY), device=DEVICE)
    u = torch.zeros((NX, NY, 2), device=DEVICE)
    
    # Inline equilibrium calculation for initialization
    cu = torch.einsum("dc,xyc->dxy", E, u)  # [9,NX,NY]
    usqr = (u**2).sum(-1)  # [NX,NY]
    f = rho * W_opt * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * usqr)  # [9,NX,NY]
    
    return f  # Returns [9,NX,NY]


def stream(f):
    """Vectorized streaming"""
    return torch.stack(
        [torch.roll(f[i], shifts=tuple(SHIFTS[i]), dims=(0, 1)) for i in range(9)]
    )


def collide_and_boundary(f):
    """Fused collision + boundary"""
    # 1. Compute macroscopic quantities
    rho = f.sum(dim=0)  # [NX,NY]
    u = torch.einsum("dc,dxy->xyc", E, f) / rho.unsqueeze(-1)

    # 2. Apply boundary conditions
    u[:, -1, 0] = LID_VELOCITY  # Moving lid
    u[:, -1, 1] = 0

    # 3. Bounce-back walls (in-place for efficiency)
    f[[2, 5, 6], :, 0] = f[[4, 7, 8], :, 0]  # Bottom
    f[[1, 5, 8], 0, :] = f[[3, 6, 7], 0, :]  # Left
    f[[3, 6, 7], -1, :] = f[[1, 5, 8], -1, :]  # Right

    # 4. Collision - equilibrium
    cu = torch.einsum("dc,xyc->dxy", E, u)  # [9,NX,NY]
    usqr = (u**2).sum(-1)  # [NX,NY]
    feq = rho * W_opt * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * usqr)  # [9,NX,NY]
    
    # In-place collision update
    f -= (1 / TAU) * (f - feq)


# ==============================================
# Main Simulation
# ==============================================


def run_simulation():
    print(" Running simulation(Milestone 5 updated) with parameters :")
    print(
        f"NX: {NX}, NY: {NY}, NSTEPS: {NSTEPS}, OMEGA: {OMEGA}, TAU: {TAU}, LID_VELOCITY: {LID_VELOCITY}"
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    f = initialize()
    
    torch.cuda.synchronize()
    start = time.time()

    for step in range(NSTEPS):
        f = stream(f)
        collide_and_boundary(f)  # In-place modification

        if step % 500 == 0 and step > 0:
            torch.cuda.synchronize()
            blups = (step * NX * NY) / (time.time() - start) / 1e9
            print(f"Step {step:5d}, BLUPS: {blups:.2f}")

    torch.cuda.synchronize()
    total_time = time.time() - start
    final_blups = (NSTEPS * NX * NY) / total_time / 1e9
    mlups = (NSTEPS * NX * NY) / total_time / 1e6
    print(f"Final BLUPS: {final_blups:.2f}")
    print(f"Final MLUPS: {mlups:.2f}")
    print(f"Total time: {total_time:.2f}s")


# ==============================================
# Benchmarking and Plotting Function
# ==============================================
def benchmark_and_plot():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Running benchmark and plotting... on DEVICE:", DEVICE, "at", timestamp)
    grid_sizes = [1000, 3000, 5000, 8000, 10000, 15000, 20000]
    # grid_sizes = [18000, 24000, 30000, 35000]  
    mlups_results, power_draws, gpu_elapsed_times = [], [], []

    # CPU baseline - Fix: Create CPU versions of tensors
    cpu_device = torch.device("cpu")
    cpu_grid = 1000
    cpu_steps = 200
    f_cpu = torch.ones((9, cpu_grid, cpu_grid), device=cpu_device) * 0.1
    
    # Create CPU versions of constants
    E_cpu = E.to(cpu_device)  # Move E to CPU
    W_cpu = W.to(cpu_device)  # Move W to CPU
    W_opt_cpu = W_cpu.reshape(9, 1, 1)  # CPU version of W_opt

    def stream_cpu(f):
        return torch.stack(
            [
                torch.roll(f[i], shifts=(int(E_cpu[i, 0]), int(E_cpu[i, 1])), dims=(0, 1))  # Use E_cpu
                for i in range(9)
            ]
        )

    def collide_cpu(f):
        rho = f.sum(0)
        u = torch.einsum("dc,dxy->xyc", E_cpu, f) / rho.unsqueeze(-1)  # Use E_cpu
        u[:, -1, 0] = LID_VELOCITY
        u[:, -1, 1] = 0
        f[[2, 5, 6], :, 0] = f[[4, 7, 8], :, 0]
        f[[1, 5, 8], 0, :] = f[[3, 6, 7], 0, :]
        f[[3, 6, 7], -1, :] = f[[1, 5, 8], -1, :]
        
        # Inline equilibrium for CPU to avoid function calls
        cu = torch.einsum("dc,xyc->dxy", E_cpu, u)
        usqr = (u**2).sum(-1)
        feq = rho * W_opt_cpu * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * usqr)
        
        f[:] -= (1 / TAU) * (f - feq)

    print("Running CPU baseline (pure PyTorch)...")
    for _ in range(10):
        f_cpu = stream_cpu(f_cpu)
        collide_cpu(f_cpu)

    start_cpu = time.time()
    for _ in range(cpu_steps):
        f_cpu = stream_cpu(f_cpu)
        collide_cpu(f_cpu)
    cpu_time = time.time() - start_cpu
    print(f"CPU baseline (1000x1000, 200 steps): {cpu_time:.2f}s")

    baseline_updates = cpu_grid * cpu_grid * cpu_steps
    for nx in grid_sizes:
        print(f"Running benchmark for grid size {nx}x{nx}...")
        steps = 2000
        f = torch.ones((9, nx, nx), device=DEVICE) * 0.1
        
        print(f"  Running {nx}x{nx} benchmark...")
            
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(steps):
            f = stream(f)
            collide_and_boundary(f)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        updates = nx * nx * steps
        mlups_results.append(updates / elapsed / 1e6)
        gpu_elapsed_times.append(elapsed)
        print(f"Grid {nx}x{nx}, Steps {steps}, MLUPS: {mlups_results[-1]:.2f}, Time: {elapsed:.2f}s")
        try:
            gpus = GPUtil.getGPUs()
            power_draws.append(gpus[0].powerDraw if gpus else 0)
        except:
            power_draws.append(0)
        import gc

        # After each benchmark run
        del f
        torch.cuda.empty_cache()
        gc.collect()

    print("Benchmarking complete. Plotting now..")
    # Calculations
    mlups_baseline = mlups_results[0]
    cpu_est_times = [
        cpu_time * ((nx * nx * 2000) / baseline_updates) for nx in grid_sizes
    ]
    speedups_vs_cpu = [ce / ge for ce, ge in zip(cpu_est_times, gpu_elapsed_times)]
    speedups_vs_smallest_gpu = [m / mlups_baseline for m in mlups_results]
    mlups_per_watt = [m / p if p > 0 else 0 for m, p in zip(mlups_results, power_draws)]

    # Plotting
    PLOT_DIR_B.mkdir(exist_ok=True, parents=True)

    def plot_graph(yvals, title, ylabel, filename, annotations=None, color="blue"):
        plt.figure()
        plt.plot(grid_sizes, yvals, marker="o", color=color)
        plt.title(title)
        plt.xlabel("Grid Size")
        plt.ylabel(ylabel)
        plt.grid(True)
        if annotations:
            for x, y in zip(grid_sizes, yvals):
                plt.annotate(
                    f"{y:.1f}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=8,
                )
        # plt.savefig(PLOT_DIR / filename)
        plt.savefig(PLOT_DIR_B / f"{timestamp}_{filename}")
        plt.close()

    plot_graph(
        mlups_results,
        "Grid Size vs MLUPS",
        "MLUPS",
        "mlups_vs_grid.png",
        annotations=True,
    )
    plot_graph(
        speedups_vs_cpu,
        "Speedup vs CPU",
        "Speedup",
        "speedup_vs_cpu.png",
        annotations=True,
        color="purple",
    )
    plot_graph(
        speedups_vs_smallest_gpu,
        "Relative GPU Speedup",
        "Speedup vs smallest GPU",
        "gpu_relative_speedup.png",
        annotations=True,
        color="orange",
    )
    plot_graph(
        mlups_per_watt,
        "Efficiency: MLUPS per Watt",
        "MLUPS/Watt",
        "efficiency_mlups_per_watt.png",
        annotations=True,
        color="green",
    )

    # Runtime comparison
    x_idx = np.arange(len(grid_sizes))
    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(x_idx - width / 2, cpu_est_times, width, label="CPU-est", color="gray")
    plt.bar(x_idx + width / 2, gpu_elapsed_times, width, label="GPU-time", color="teal")
    plt.xticks(x_idx, grid_sizes)
    plt.title("CPU vs GPU Runtime")
    plt.ylabel("Time (s)")
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR_B / "cpu_vs_gpu_runtime.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark plots")
    parser.add_argument("--max-autotune", action="store_true", help="Use max-autotune compilation mode (may be unstable)")
    args = parser.parse_args()

    if args.benchmark:
        benchmark_and_plot()
    else:
        run_simulation()
