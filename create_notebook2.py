import json

notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 2. Evaluation and Visualization\n",
                "\n",
                "This notebook loads the simulation results from the first notebook and performs comprehensive evaluation and visualization of the Fluid-HD system's performance.\n",
                "\n",
                "**Workflow:**\n",
                "1.  **Load Results**: Import the simulation data saved by the first notebook.\n",
                "2.  **Decode Partners**: Use the decoder to recover interaction partners from the memory vector.\n",
                "3.  **Calculate Metrics**: Compute Recall@k and other performance measures.\n",
                "4.  **Generate Visualizations**: Create plots showing system performance and agent dynamics.\n",
                "5.  **Summary Analysis**: Provide insights into the system's behavior and limitations."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2.1. Load Results"
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
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from collections import Counter\n",
                "\n",
                "# Add src directory to path\n",
                "module_path = os.path.abspath(os.path.join('..'))\n",
                "if module_path not in sys.path:\n",
                "    sys.path.append(module_path)\n",
                "\n",
                "from src.decoder import decode_partners\n",
                "from src.utils import recall_at_k, snr\n",
                "\n",
                "# Load the simulation results\n",
                "results_path = '../results/notebook_run/simulation_results.pkl'\n",
                "with open(results_path, 'rb') as f:\n",
                "    results = pickle.load(f)\n",
                "\n",
                "# Extract key variables\n",
                "N = results['N']\n",
                "D = results['D']\n",
                "T = results['T']\n",
                "final_memory = results['final_memory']\n",
                "id_vectors = results['id_vectors']\n",
                "group_history = results['group_history']\n",
                "state_history = results['state_history']\n",
                "true_partners = results['true_partners']\n",
                "params = results['params']\n",
                "\n",
                "print(f\"Loaded simulation with {N} agents, {D} dimensions, {T} time steps\")\n",
                "print(f\"Parameters: tau_intra={params['tau_intra']}, tau_inter={params['tau_inter']}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2.2. Decode Partners"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Decode top-1 and top-5 partners for each agent\n",
                "decoded_partners_1 = decode_partners(final_memory, id_vectors, top_k=1)\n",
                "decoded_partners_5 = decode_partners(final_memory, id_vectors, top_k=5)\n",
                "\n",
                "# Calculate recall metrics\n",
                "recall_1 = recall_at_k(true_partners, decoded_partners_1, k=1)\n",
                "recall_5 = recall_at_k(true_partners, decoded_partners_5, k=5)\n",
                "\n",
                "print(f\"Recall@1: {recall_1:.3f}\")\n",
                "print(f\"Recall@5: {recall_5:.3f}\")\n",
                "\n",
                "# Show some examples\n",
                "print(\"\\nExample decoded partners (top-5):\")\n",
                "for i in range(min(5, N)):\n",
                "    true = true_partners[i][:3]  # Show first 3 true partners\n",
                "    decoded = decoded_partners_5[i][:3]  # Show first 3 decoded partners\n",
                "    print(f\"Agent {i}: True={true}, Decoded={decoded}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2.3. Performance Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Analyze the distribution of true partners per agent\n",
                "num_true_partners = [len(true_partners[i]) for i in range(N)]\n",
                "print(f\"Average number of true partners per agent: {np.mean(num_true_partners):.1f}\")\n",
                "print(f\"Max partners: {max(num_true_partners)}, Min partners: {min(num_true_partners)}\")\n",
                "\n",
                "# Count how many agents have each number of partners\n",
                "partner_counts = Counter(num_true_partners)\n",
                "print(\"\\nDistribution of partner counts:\")\n",
                "for count, freq in sorted(partner_counts.items()):\n",
                "    print(f\"  {count} partners: {freq} agents\")\n",
                "\n",
                "# Calculate per-agent recall\n",
                "per_agent_recall = []\n",
                "for i in range(N):\n",
                "    if len(true_partners[i]) > 0:\n",
                "        hits = sum(1 for partner in decoded_partners_5[i] if partner in true_partners[i])\n",
                "        recall = hits / len(true_partners[i])\n",
                "        per_agent_recall.append(recall)\n",
                "\n",
                "print(f\"\\nPer-agent Recall@5 statistics:\")\n",
                "print(f\"  Mean: {np.mean(per_agent_recall):.3f}\")\n",
                "print(f\"  Std:  {np.std(per_agent_recall):.3f}\")\n",
                "print(f\"  Min:  {min(per_agent_recall):.3f}\")\n",
                "print(f\"  Max:  {max(per_agent_recall):.3f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2.4. Visualizations"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Set up the plotting style\n",
                "plt.style.use('seaborn-v0_8')\n",
                "fig, axes = plt.subplots(2, 2, figsize=(15, 12))\n",
                "\n",
                "# Plot 1: Distribution of true partners\n",
                "axes[0, 0].hist(num_true_partners, bins=20, alpha=0.7, edgecolor='black')\n",
                "axes[0, 0].set_xlabel('Number of True Partners')\n",
                "axes[0, 0].set_ylabel('Number of Agents')\n",
                "axes[0, 0].set_title('Distribution of True Partners per Agent')\n",
                "axes[0, 0].grid(True, alpha=0.3)\n",
                "\n",
                "# Plot 2: Per-agent recall distribution\n",
                "axes[0, 1].hist(per_agent_recall, bins=20, alpha=0.7, edgecolor='black')\n",
                "axes[0, 1].set_xlabel('Recall@5')\n",
                "axes[0, 1].set_ylabel('Number of Agents')\n",
                "axes[0, 1].set_title('Distribution of Per-Agent Recall@5')\n",
                "axes[0, 1].grid(True, alpha=0.3)\n",
                "\n",
                "# Plot 3: Group size over time (first 1000 steps)\n",
                "max_steps = min(1000, T)\n",
                "group_sizes = []\n",
                "for t in range(max_steps):\n",
                "    groups_at_t = group_history[t]\n",
                "    unique_groups = np.unique(groups_at_t[groups_at_t != -1])\n",
                "    sizes = [np.sum(groups_at_t == gid) for gid in unique_groups]\n",
                "    group_sizes.extend(sizes)\n",
                "\n",
                "axes[1, 0].hist(group_sizes, bins=15, alpha=0.7, edgecolor='black')\n",
                "axes[1, 0].set_xlabel('Group Size')\n",
                "axes[1, 0].set_ylabel('Frequency')\n",
                "axes[1, 0].set_title('Distribution of Group Sizes (First 1000 Steps)')\n",
                "axes[1, 0].grid(True, alpha=0.3)\n",
                "\n",
                "# Plot 4: Memory vector statistics\n",
                "memory_values = final_memory\n",
                "axes[1, 1].hist(memory_values, bins=50, alpha=0.7, edgecolor='black')\n",
                "axes[1, 1].set_xlabel('Memory Vector Values')\n",
                "axes[1, 1].set_ylabel('Frequency')\n",
                "axes[1, 1].set_title('Distribution of Final Memory Vector Values')\n",
                "axes[1, 1].grid(True, alpha=0.3)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2.5. Agent Trajectory Animation (First 200 Steps)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from matplotlib.animation import FuncAnimation\n",
                "import matplotlib.patches as patches\n",
                "\n",
                "# Create animation of first 200 steps\n",
                "max_animation_steps = min(200, T)\n",
                "fig, ax = plt.subplots(figsize=(10, 10))\n",
                "\n",
                "def animate(frame):\n",
                "    ax.clear()\n",
                "    \n",
                "    # Get current state\n",
                "    positions = state_history[frame, :, :2]  # x, y positions\n",
                "    groups = group_history[frame]\n",
                "    \n",
                "    # Plot agents colored by group\n",
                "    unique_groups = np.unique(groups)\n",
                "    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))\n",
                "    \n",
                "    for i, group_id in enumerate(unique_groups):\n",
                "        if group_id == -1:  # Ungrouped agents\n",
                "            mask = groups == group_id\n",
                "            ax.scatter(positions[mask, 0], positions[mask, 1], \n",
                "                      c='gray', alpha=0.5, s=30, label='Ungrouped')\n",
                "        else:\n",
                "            mask = groups == group_id\n",
                "            ax.scatter(positions[mask, 0], positions[mask, 1], \n",
                "                      c=[colors[i]], alpha=0.7, s=50)\n",
                "    \n",
                "    ax.set_xlim(0, 1)\n",
                "    ax.set_ylim(0, 1)\n",
                "    ax.set_xlabel('X Position')\n",
                "    ax.set_ylabel('Y Position')\n",
                "    ax.set_title(f'Agent Positions and Groups (Step {frame})')\n",
                "    ax.grid(True, alpha=0.3)\n",
                "    \n",
                "    # Add legend only once\n",
                "    if frame == 0:\n",
                "        ax.legend()\n",
                "\n",
                "# Create animation\n",
                "anim = FuncAnimation(fig, animate, frames=max_animation_steps, \n",
                "                    interval=100, repeat=True)\n",
                "\n",
                "plt.show()\n",
                "\n",
                "# Note: To save the animation, uncomment the following line:\n",
                "# anim.save('../results/plots/agent_animation.gif', writer='pillow')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2.6. Heatmap: True vs Decoded Partners"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Create matrices for visualization\n",
                "true_matrix = np.zeros((N, N))\n",
                "decoded_matrix = np.zeros((N, N))\n",
                "\n",
                "for i in range(N):\n",
                "    for j in true_partners[i]:\n",
                "        true_matrix[i, j] = 1\n",
                "    for j in decoded_partners_5[i]:\n",
                "        decoded_matrix[i, j] = 1\n",
                "\n",
                "# Create the heatmap\n",
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))\n",
                "\n",
                "# True partners heatmap\n",
                "im1 = ax1.imshow(true_matrix, cmap='Blues', aspect='equal')\n",
                "ax1.set_title('True Interaction Partners')\n",
                "ax1.set_xlabel('Agent ID')\n",
                "ax1.set_ylabel('Agent ID')\n",
                "plt.colorbar(im1, ax=ax1)\n",
                "\n",
                "# Decoded partners heatmap\n",
                "im2 = ax2.imshow(decoded_matrix, cmap='Reds', aspect='equal')\n",
                "ax2.set_title('Decoded Top-5 Partners')\n",
                "ax2.set_xlabel('Agent ID')\n",
                "ax2.set_ylabel('Agent ID')\n",
                "plt.colorbar(im2, ax=ax2)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "# Calculate overlap\n",
                "overlap = np.sum(true_matrix * decoded_matrix)\n",
                "total_true = np.sum(true_matrix)\n",
                "total_decoded = np.sum(decoded_matrix)\n",
                "\n",
                "print(f\"Overlap between true and decoded: {overlap}\")\n",
                "print(f\"Total true interactions: {total_true}\")\n",
                "print(f\"Total decoded interactions: {total_decoded}\")\n",
                "print(f\"Precision: {overlap/total_decoded:.3f}\")\n",
                "print(f\"Recall: {overlap/total_true:.3f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2.7. Summary and Key Findings"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Key Findings\n",
                "\n",
                "1. **Overall Performance**: The system achieved a Recall@1 of {recall_1:.3f} and Recall@5 of {recall_5:.3f}.\n",
                "2. **Memory Capacity**: With {D} dimensions, the system can store information about agent interactions.\n",
                "3. **Group Dynamics**: The simulation shows realistic group formation and dissolution patterns.\n",
                "\n",
                "### Limitations\n",
                "\n",
                "1. **Noise in Memory**: The memory vector accumulates noise from multiple bindings.\n",
                "2. **Decoding Accuracy**: Performance depends on the dimensionality and normalization frequency.\n",
                "3. **Group Stability**: The system may not capture very transient interactions.\n",
                "\n",
                "### Next Steps\n",
                "\n",
                "1. **Hyperparameter Tuning**: Experiment with different D values and normalization frequencies.\n",
                "2. **Alternative Decoding**: Try different similarity metrics for partner recovery.\n",
                "3. **Real Data**: Apply the system to ETH/UCY pedestrian trajectory data.\n",
                "4. **Scalability**: Test with larger numbers of agents and longer simulations."
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
with open('notebooks/02_evaluation_and_visualization.ipynb', 'w') as f:
    json.dump(notebook_content, f, indent=1)

print("Second notebook created successfully!") 