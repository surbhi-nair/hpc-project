# Project: Lattice Boltzmann Method for Fluid Dynamics Simulation
Project as part of the course "High-performance computing: Distributed-memory parallelization on GPUs and accelerators" using PyTorch.

## Description
This project implements a Lattice Boltzmann Method (LBM) for simulating fluid dynamics
using high-performance computing techniques. The code is designed to run with GPU acceleration and is optimized for performance and scalability.


## Milestones:
### Milestone 1: Project Structure
- `src/`: Contains the source code for the LBM implementation.
    - `src/milestones/`: Contains milestone-specific implementations and optimizations.
    - `src/utils/`: Contains utility functions for data handling and visualization.
    - `src/sliding_lid/`: Contains the sliding lid implementation of the LBM in a modular and structured way.
- `plots/`: Contains the plots generated from the simulation results.


### Milestone 2: Streaming Operator
- Refer the file milestone2.py for the implementation of the streaming operator.
- Command to run: `python src/milestones/milestone2.py`
- This will execute the streaming operator of the LBM and generate plots for the simulation results in the `plots/m2` directory.
- Observations:
    - For velocity plots, we observe a single right-pointing arrow that travels to the right, indicating that the particles are moving in the east direction starting from the center of the grid.
    - For density plots, we observe a square region in the center of the grid that moves to the right, indicating that the density of particles is concentrated in that region and is also moving in the east direction.
- Optionally, you can run `python src/utils/run_animations.py --milestone 2` to create animations(gifs) of the density and velocity plots previously generated. These will also be saved in the `plots/milestone/m2` directory.

### Milestone 3: Collision Operator
- Refer the file milestone3.py for the implementation of the collision operator.
- Command to run: `python src/milestones/milestone3.py`
- This will execute the collision operator of the LBM and generate plots for the simulation results in the `plots/m3` directory for each of the test cases.
- The test cases include:
    - Test Case 1: Initially, uniform density on the grid with a small perturbation(higher density value) in the center.
    - Test Case 2: Choosing an initial distribution of $\rho(r)$ and $u(r)$ at $t = 0$
- Test Case 1 Observations:
    -  We choose density to be 0.5 everywhere except for a square region in the center where it is 0.9.
    - We observe how a static system reacts to a density perturbation i.e. how density gradients induce velocity (via collisions → momentum transfer)
<!-- 

    | Time Step | Density Plot | Velocity Plot |
    |-----------|--------------|----------------|
    | 0         | A square region in the center with higher density | Velocity vectors pointing outwards from the center (Although $u$ is initialised 0 but because of the non-uniform density, the equilibrium distribution f_eq causes directional imbalance in the momentum field) |
    | 1 - 30 | Central bump diffuses | As system relaxes, velocity vectors become shorter — the system is losing net momentum (due to collisions)|
    | 30 - 50 | The square region starts to spread out, indicating diffusion of density | Velocity vectors are very short across the grid. The flow field approaches steady state ($t \rightarrow \infty$). Eventually we can say that, all fluid elements settle with negligible motion — the system reaches near-zero velocity everywhere. Note that due to round-off effects or slight residual imbalances, the remaining tiny velocities might show very weak “restorative” movement toward the center. This happens because the system “overshot” slightly during its relaxation — it’s now gently oscillating around equilibrium| -->
<!-- 
- Test Case 2 Observations:
    - We choose initial density to be uniform everywhere and the center to have a small bump in velocity pointing to the right.
    - We observe how momentum diffuses in a static fluid with no density imbalance i.e. how a given initial velocity decays (viscous diffusion of momentum).

    | Time Step | Density Plot | Velocity Plot |
    |-----------|--------------|----------------|
    | 0         | Center has a slightly higher density towards the right | Only center has small rightward velocity; all other cells have zero velocity  |
    | 1 - 30 | Central density spreads outward symmetrically, forming waves | Velocity vectors radiate outward, then begin to curve and swirl slightly; some arrows start weakening|
    | 30 - 50 | Density field becomes almost uniform (small ripples may remain) | Velocity arrows become shorter, indicating decay; directions randomize slightly as equilibrium is approached| -->
- Optionally, you can run `python src/utils/run_animations.py --milestone 3` to create animations(gifs) of the density and velocity plots previously generated for both the test cases. These will be saved in the `plots/milestone/m3_test1` and `plots/milestone/m3_test2` directories.

#### Comparison of Test Cases 1 and 2
| Observation Point| Test 1 | Test 2 |
|------------------|---------|---------|
| **Velocity Direction (early)** | Outward vectors from center (as fluid “escapes” high-density region)   | Outward vectors from center due to initial push               |
| **Velocity Origin**         | Emerges from density imbalance (no explicit velocity initialized)      | Comes from explicit momentum initialization                   |
| **Wave Pattern in Density** | Wave-like propagation, symmetric spreading from bump                   | Density remains flat (mostly), minor induced fluctuations     |
| **Velocity Magnitude Decay**| Slightly slower decay, as momentum is built up over several steps      | Decays steadily from the start due to no further external driving |
| **Steady-State Behavior**   | Becomes symmetric equilibrium with smooth density field                | Approaches full rest state with uniform density and near-zero velocities |


### Milestone 4: Shear-wave Decay
- Refer the file milestone4.py
- Alternatively, refer the file sliding_lid/shear-wave-decay.py file for a scalable implementation of the shear-wave decay that includes a structured and abstract way to implement the script.
- Different omega values can be passed to check the theoretical and numerical viscosity being calculated

### Milestone 5: Lid-driven Cavity
- Refer the file milestone5.py for the most optimized implementation of the lid-driven cavity
- The upper wall is set to move with a velocity of 0.1 in the x-direction, while the other walls are stationary.
- The simulation was tested on the H100 GPU on the bwunicluster to observe the performance in terms of MLUPS i.e. Million Lattice updates per second.
- $\text{MLUPS} = \frac{N_x \cdot N_y \cdot \text{steps}}{\text{time (s)} \cdot 10^6}$
    - where NX and NY are the grid dimensions and steps is the number of time steps simulated.
- The code to implement the lid-driven cavity and observe the velocity profile is in the `src/milestones/milestone5.py` file.
- The *final optimized code* is in the `src/milestones/milestone5-optimized.py` file.
    - Command to run: `python src/milestones/milestone5-optimized.py`
    - This code has been optimized for performance on the bwunicluster using the H100 GPU.
    - It includes a single kernel streaming function and a fused collision and boundary function to minimize kernel launches and memory writes.
    - It also includes a benchmark function to measure the performance of the simulation and plot the results.


### Performance Benchmarks
- The performance was tested on the bwunicluster using the H100 and A100 GPUs for various grid sizes - 
    - 1000, 3000, 5000, 8000, 10000, 15000, 18000.
- The benchmark plotting functions are included in the `src/sliding_lid/plot_benchm.py` file and the plots are included in the `plots/benchmarks` directory.
- The benchmarks include both the MLUPS and BLUPS (Billion Lattice Updates per Second) for different grid sizes.
- Optionally, there is a script `run_batchjob.sh` to run the benchmark on the bwunicluster. 
    - Command to run on the bwunicluster: `sbatch run_batchjob.sh`


### Additional Notes
- Apart from the milestone-specific implementations, the `src/sliding_lid` directory contains a structured and abstract way to implement the LBM simulation. This allows for easier experimentation and modification of the code as it allows the parameters to be passed as arguments while running the code and easier modification of the script. 
- However, for the purpose of the final code implementation, the milestone-specific implementations are used as they are more optimized and tailored for the specific use case.
