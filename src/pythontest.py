import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
import time

def run():
    print("Inside run function of pythontest.py")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

if __name__ == "__main__":
    print("Inside main block of pythontest.py")
    run()