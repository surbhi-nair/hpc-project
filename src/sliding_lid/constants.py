import numpy as np
import torch
import os
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# D2Q9 velocities [9 directions, xy coordinates] shape (9x2)
CHANNEL_VELOCITIES = torch.tensor(
    [[0, 1, 0, -1, 0, 1, -1, -1, 1], [0, 0, 1, 0, -1, 1, 1, -1, -1]],
    dtype=torch.int32,
    device=DEVICE,
).T

# Weights for each direction
WEIGHTS = torch.tensor(
    [
        4.0 / 9,
        1.0 / 9,
        1.0 / 9,
        1.0 / 9,
        1.0 / 9,
        1.0 / 36,
        1.0 / 36,
        1.0 / 36,
        1.0 / 36,
    ],
    dtype=torch.float32,
    device=DEVICE,
)


PLOT_FLAG = False  # Set to True to enable plotting

# Shear wave decay
SW_EPSILON = 0.05

# D2Q9 channels and anti-channels
channels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
anti_channels = [0, 3, 4, 1, 2, 7, 8, 5, 6]

# Up, Down, Left, Right channels for boundary conditions
up_in_channels = (4, 7, 8)
up_out_channels = (2, 5, 6)

down_in_channels = (2, 5, 6)
down_out_channels = (4, 7, 8)

left_in_channels = (1, 5, 8)
left_out_channels = (3, 7, 6)

right_in_channels = (3, 7, 6)
right_out_channels = (1, 5, 8)

c_s_squared = torch.tensor(1 / 3.0, dtype=torch.float32, device=DEVICE)
