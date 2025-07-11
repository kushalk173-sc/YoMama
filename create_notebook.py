import json

notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 1. Simulation and Binding\n",
                "\n",
                "This notebook runs the core agent-based simulation and performs hyperdimensional binding to record agent interactions. It serves as the first step in the Fluid-HD experiment pipeline.\n",
                "\n",
                "**Workflow:**\n",
                "1.  **Setup & Imports**: Load necessary libraries and add the `src` directory to the path.\n",
                "2.  **Set Hyperparameters**: Define the parameters for this specific simulation run (e.g., N, D, T).\n",
                "3.  **Instantiate Core Components**: Create instances of the `AgentSimulator`, `HDBinder`, and `MemoryStore`.\n",
                "4.  **Run Simulation Loop**: Iterate through time steps, updating agent states, determining groups, and binding co-occurrences into the memory vector.\n",
                "5.  **Save Results**: Store the outputs (final memory, group history, true partner list) to disk for the next notebook."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1.1. Setup & Imports"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import numpy as np\n",
                "import pandas as pd\n",
                "import os\n",
                "import sys\n",
                "import pickle\n",
                "from tqdm.notebook import tqdm\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "# Add src directory to path to import our modules\n",
                "module_path = os.path.abspath(os.path.join('..'))\n",
                "if module_path not in sys.path:\n",
                "    sys.path.append(module_path)\n",
                "\n",
                "from src.simulation import AgentSimulator\n",
                "from src.affinity import compute_hd_embeddings, cosine_affinity\n",
                "from src.grouping import build_adjacency, extract_groups\n",
                "from src.binding import HDBinder\n",
                "from src.memory import MemoryStore\n",
                "from src.utils import set_seed"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1.2. Set Hyperparameters\n",
                "\n",
                "Here, we define the key parameters for our simulation run. These can be tuned to explore different dynamics."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "N = 100                  # Number of agents\n",
                "D = 4000                 # Dimensionality of hypervectors\n",
                "T = 5000                 # Total simulation time steps\n",
                "SEED = 42                # Random seed for reproducibility\n",
                "\n",
                "# Grouping parameters\n",
                "TAU_INTRA = 1.2          # Multiplier for intra-group affinity\n",
                "TAU_INTER = 0.8          # Multiplier for inter-group affinity\n",
                "AFFINITY_THRESHOLD = 0.6 # Cosine similarity threshold for an edge\n",
                "\n",
                "# Output directory\n",
                "OUTPUT_DIR = '../results/notebook_run/'\n",
                "os.makedirs(OUTPUT_DIR, exist_ok=True)\n",
                "\n",
                "set_seed(SEED)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1.3. Instantiate Core Components"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "simulator = AgentSimulator(N=N, seed=SEED)\n",
                "binder = HDBinder(N=N, D=D, seed=SEED)\n",
                "memory_store = MemoryStore(D=D)\n",
                "\n",
                "# Random projection matrix for converting agent states to HD vectors\n",
                "state_dim = simulator.state.shape[1] # e.g., 4 for (x, y, vx, vy)\n",
                "W_proj = np.random.randn(D, state_dim)\n",
                "\n",
                "print(f\"Initialized simulation with {N} agents.\")\n",
                "print(f\"HD vectors dimensionality: {D}\")\n",
                "print(f\"State vector dimensionality: {state_dim}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1.4. Run Simulation Loop"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "group_history = []\n",
                "state_history = []\n",
                "prev_groups = np.zeros(N, dtype=int)\n",
                "\n",
                "for t in tqdm(range(T), desc=\"Simulating\"):\n",
                "    # 1. Get current agent states\n",
                "    states = simulator.step()\n",
                "    state_history.append(states)\n",
                "    \n",
                "    # 2. Determine groups\n",
                "    hd_embeddings = compute_hd_embeddings(states, W_proj)\n",
                "    affinity_matrix = cosine_affinity(hd_embeddings, threshold=AFFINITY_THRESHOLD)\n",
                "    adj_matrix = build_adjacency(affinity_matrix, prev_groups, TAU_INTRA, TAU_INTER)\n",
                "    current_groups = extract_groups(adj_matrix)\n",
                "    \n",
                "    # 3. Bind co-occurring agents into memory\n",
                "    unique_group_ids = np.unique(current_groups[current_groups != -1])\n",
                "    for group_id in unique_group_ids:\n",
                "        agent_indices = np.where(current_groups == group_id)[0].tolist()\n",
                "        if len(agent_indices) > 1:\n",
                "            binder.record_co_occurrence(memory_store, agent_indices)\n",
                "            \n",
                "    # 4. Periodic normalization of the memory vector\n",
                "    if t > 0 and t % 200 == 0:\n",
                "        memory_store.normalize()\n",
                "    \n",
                "    # 5. Store history and update previous groups\n",
                "    group_history.append(current_groups)\n",
                "    prev_groups = current_groups\n",
                "\n",
                "# Final normalization\n",
                "memory_store.normalize()\n",
                "\n",
                "group_history = np.array(group_history)\n",
                "state_history = np.array(state_history)\n",
                "final_memory = memory_store.get_memory()\n",
                "id_vectors = binder.E_id\n",
                "\n",
                "print(\"Simulation complete.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Quick Visualization: Agent Trajectories\n",
                "\n",
                "Let's plot the paths of the first few agents to see what the simulation looks like."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(8, 8))\n",
                "for i in range(min(N, 10)):\n",
                "    plt.plot(state_history[:, i, 0], state_history[:, i, 1], alpha=0.7)\n",
                "plt.title('Trajectories of First 10 Agents')\n",
                "plt.xlabel('X Position')\n",
                "plt.ylabel('Y Position')\n",
                "plt.xlim(0, 1)\n",
                "plt.ylim(0, 1)\n",
                "plt.grid(True)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1.5. Save Results\n",
                "\n",
                "We now save the key artifacts to disk. These will be loaded by the second notebook for evaluation and visualization."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Generate the ground truth for evaluation\n",
                "true_partners = {i: set() for i in range(N)}\n",
                "for groups_at_t in group_history:\n",
                "    for group_id in np.unique(groups_at_t[groups_at_t != -1]):\n",
                "        members = np.where(groups_at_t == group_id)[0]\n",
                "        for i1 in range(len(members)):\n",
                "            for i2 in range(i1 + 1, len(members)):\n",
                "                u, v = members[i1], members[i2]\n",
                "                true_partners[u].add(v)\n",
                "                true_partners[v].add(u)\n",
                "true_partners = {k: list(v) for k, v in true_partners.items()}\n",
                "\n",
                "# Package results into a dictionary\n",
                "results = {\n",
                "    'N': N,\n",
                "    'D': D,\n",
                "    'T': T,\n",
                "    'seed': SEED,\n",
                "    'final_memory': final_memory,\n",
                "    'id_vectors': id_vectors,\n",
                "    'group_history': group_history,\n",
                "    'state_history': state_history,\n",
                "    'true_partners': true_partners,\n",
                "    'params': {\n",
                "        'tau_intra': TAU_INTRA,\n",
                "        'tau_inter': TAU_INTER,\n",
                "        'affinity_threshold': AFFINITY_THRESHOLD\n",
                "    }\n",
                "}\n",
                "\n",
                "# Save to file\n",
                "output_path = os.path.join(OUTPUT_DIR, 'simulation_results.pkl')\n",
                "with open(output_path, 'wb') as f:\n",
                "    pickle.dump(results, f)\n",
                "\n",
                "print(f\"Results saved to: {output_path}\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write the notebook to file
with open('notebooks/01_simulation_and_binding.ipynb', 'w') as f:
    json.dump(notebook_content, f, indent=1)

print("Notebook created successfully!") 