"""Global variables"""
import numpy as np
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hardcoded velocity channels, shape (9x2)
velocity_channels = torch.tensor([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                                  [0, 0, 1, 0, -1, 1, 1, -1, -1]], dtype=torch.int32, device=DEVICE).T

# Small positive constant to avoid division by zero
epsilon = 1e-16

# Equilibrium occupation numbers
weights = torch.tensor([4. / 9, 1. / 9, 1. / 9, 1. / 9, 1. / 9,
                        1. / 36, 1. / 36, 1. / 36, 1. / 36], dtype=torch.float32, device=DEVICE)

# Shear wave decay constants
SW_EPSILON = 0.05

# D2Q9 channels and anti-channels
channels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
anti_channels = [0, 3, 4, 1, 2, 7, 8, 5, 6]

# Useful channels for boundary conditions. Ordering matters.
up_in_channels = (4, 7, 8)
up_out_channels = (2, 5, 6)

down_in_channels = (2, 5, 6)
down_out_channels = (4, 7, 8)

left_in_channels = (1, 5, 8)
left_out_channels = (3, 7, 6)

right_in_channels = (3, 7, 6)
right_out_channels = (1, 5, 8)

# Speed of sound squared
c_s_squared = torch.tensor(1 / 3., dtype=torch.float32, device=DEVICE)
