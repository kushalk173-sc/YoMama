# Fluid-HD Experiment

This project implements a "fluid nightclub" agent simulation to study social grouping dynamics. The simulation is instrumented with Hyperdimensional Computing (HDC) binding to create a memory of inter-agent interactions ("hops" between groups), which can later be decoded and analyzed.

## Overview

The core of the project consists of:
1.  A multi-agent simulation where agents move in a 2D space, forming and reforming groups based on proximity and social affinity.
2.  An HDC system that projects agent states into high-dimensional vectors.
3.  A binding mechanism (`HDBinder`) that records when two agents co-occur in a group by binding their ID vectors. These bindings are bundled into a single memory vector.
4.  A decoding process to query the memory and recover the most frequent interaction partners for each agent.
5.  A suite of analysis tools, tests, and visualizations to evaluate the system's performance.

## Installation

To set up the project, you need Conda.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd YoMamaSoFat32
    ```

2.  **Create the Conda environment:**
    ```bash
    conda env create -f environment.yml
    ```

3.  **Activate the environment:**
    ```bash
    conda activate fluid_hd
    ```

Alternatively, you can use pip with the `requirements.txt` file, though the Conda environment is recommended for reproducibility.

```bash
pip install -r requirements.txt
```

## How to Run

### Jupyter Notebooks

The primary workflow is managed through two Jupyter notebooks in the `notebooks/` directory.

1.  `01_simulation_and_binding.ipynb`: Run this notebook to execute the agent simulation, perform the HDC binding, and save the results (hop logs, memory vectors) to the `results/` directory.
2.  `02_evaluation_and_visualization.ipynb`: Run this notebook to load the simulation results, decode the interaction memory, calculate performance metrics (e.g., Recall@k), and generate plots and visualizations.

### Experiment Script

For running hyperparameter sweeps, use the `run_experiments.py` script located in the `scripts/` directory.

```bash
python scripts/run_experiments.py --N 100 --T 5000 --D_list 1000 2000 --tau_intra_list 0.8 0.9 --output_dir results/sweeps
```

This script allows for programmatic exploration of different parameters and saves the outputs in a structured way.

## Results Summary

*(This section should be filled in after running the experiments.)*

Key findings from the analysis will be summarized here. This includes plots showing:
*   Recall@k vs. hyperdimensional vector dimensionality (D).
*   Recall@k vs. grouping affinity thresholds (τ_intra, τ_inter).
*   Signal-to-Noise Ratio (SNR) for decoded pairs.
*   Heatmaps comparing true vs. decoded interaction counts.

An animation of the agent simulation will also be generated and linked here.
#   Y o M a m a 
 
 
