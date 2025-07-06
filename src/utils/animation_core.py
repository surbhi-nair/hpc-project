# src/utils/animation_core.py

import imageio
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
import json

def create_animation(image_dir, prefix, output_gif, duration=2.0):
    """
    Creates a GIF from PNG images in the specified directory with a common prefix.

    Args:
        image_dir (str or Path): Path to directory containing PNG files.
        prefix (str): File prefix to match (e.g., 'm3_density_step_').
        output_gif (str): Name of the output GIF file (e.g., 'm3_density.gif').
        duration (float): Time in seconds per frame.
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory {image_dir} does not exist")

    pattern = re.compile(f"{re.escape(prefix)}\\d+\\.png")
    image_files = sorted([
        f for f in image_dir.iterdir()
        if f.is_file() and pattern.match(f.name)
    ])

    if not image_files:
        raise ValueError(f"No matching PNG files found in '{image_dir}' for prefix '{prefix}'")

    print(f"[INFO] Creating animation with {len(image_files)} frames: {output_gif}")
    images = [imageio.imread(f) for f in image_files]
    output_path = image_dir / output_gif
    imageio.mimsave(output_path, images, duration=duration) # add loop =0 for infinite loop in gif
    print(f"[SUCCESS] Saved animation: {output_path}")

def plot_velocity_decay_with_steady_state(test_case, threshold=1e-5):
    """
    Reads velocity magnitudes JSON and plots decay with steady-state detection.
    """
    base_dir = Path("plots")/f"m3_{test_case}"
    data = json.loads((base_dir/"velocity_magnitudes.json").read_text())
    vel = np.array(data)
    deltas = np.abs(np.diff(vel))
    # detect first index where change < threshold
    idx = np.argmax(deltas < threshold)
    # if never below threshold, mark end
    if not np.any(deltas < threshold):
        idx = len(vel)-1

    plt.figure(figsize=(8,4))
    plt.plot(vel, label="Mean |u|", color="blue")
    plt.axvline(idx, color="red", linestyle="--",
                label=f"Steady at step {idx}")
    plt.xlabel("Timestep")
    plt.ylabel("Mean Velocity Magnitude")
    plt.title(f"Velocity Decay & Steady-State Detection ({test_case})")
    plt.legend(); plt.grid(True); plt.tight_layout()

    out = base_dir/f"velocity_decay_ss.png"
    plt.savefig(out)
    plt.close()
    print("Saved plot to", out)
