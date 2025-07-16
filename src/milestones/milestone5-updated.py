import torch
import time
from pathlib import Path

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
NX, NY = 3000, 3000  # Grid size
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
    print(f"Final BLUPS: {final_blups:.2f}")

if __name__ == "__main__":
    run_simulation()