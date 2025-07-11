import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
import time
import sys
print("=== DEBUG START ===", file=sys.stderr)  # Force to stderr
print(f"Python version: {sys.version}", file=sys.stderr)
print(f"PyTorch CUDA available: {torch.cuda.is_available()}", file=sys.stderr)

def run():
    print("=== ENTERED RUN ===")  # Add more prints
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}", flush=True)  # Force immediate output

if __name__ == "__main__":
    print("=== MAIN BLOCK ===", flush=True)
    run()
