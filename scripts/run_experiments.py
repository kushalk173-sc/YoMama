import argparse
import numpy as np
import pandas as pd
import os
import json
import pickle
from tqdm import tqdm
import itertools

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.simulation import AgentSimulator
from src.affinity import compute_hd_embeddings, cosine_affinity
from src.grouping import build_adjacency, extract_groups
from src.binding import HDBinder
from src.memory import MemoryStore
from src.decoder import decode_partners
from src.utils import set_seed, recall_at_k

def run_simulation(N, T, D, tau_intra, tau_inter, seed, affinity_threshold=0.5):
    """Main simulation logic."""
    set_seed(seed)

    # 1. Initialization
    simulator = AgentSimulator(N=N, seed=seed)
    binder = HDBinder(N=N, D=D, seed=seed)
    memory_store = MemoryStore(D=D)
    
    # HD projection matrix for agent states
    state_dim = simulator.state.shape[1]
    W_proj = np.random.randn(D, state_dim)

    hop_log = {} # {(t, agent_id): (old_group, new_group)}
    group_history = []
    
    prev_groups = np.zeros(N, dtype=int)

    # 2. Simulation Loop
    for t in tqdm(range(T), desc="Simulating"):
        states = simulator.step()
        
        # 3. Grouping Logic
        hd_embeddings = compute_hd_embeddings(states, W_proj)
        affinity_matrix = cosine_affinity(hd_embeddings, threshold=affinity_threshold)
        adj_matrix = build_adjacency(affinity_matrix, prev_groups, tau_intra, tau_inter)
        current_groups = extract_groups(adj_matrix)
        
        # 4. Detect hops and update hop_log
        for i in range(N):
            if current_groups[i] != prev_groups[i] and t > 0:
                hop_log[(t, i)] = (prev_groups[i], current_groups[i])
        
        # 5. Binding and Bundling
        # Find all unique groups and record co-occurrences
        unique_groups = pd.Series(current_groups).unique()
        for group_id in unique_groups:
            if group_id == -1: continue # Skip ungrouped agents if any
            agent_indices_in_group = np.where(current_groups == group_id)[0].tolist()
            if len(agent_indices_in_group) > 1:
                binder.record_co_occurrence(memory_store, agent_indices_in_group)
        
        # Periodic normalization
        if t > 0 and t % 100 == 0:
            memory_store.normalize()

        prev_groups = current_groups
        group_history.append(current_groups)

    memory_store.normalize() # Final normalization
    
    # 6. Post-processing: Generate true partner list from hop_log (simplified)
    # A true pair is two agents who were in the same group at any point.
    true_partners = {i: set() for i in range(N)}
    for groups in group_history:
        for group_id in np.unique(groups):
            members = np.where(groups == group_id)[0]
            for i in members:
                for j in members:
                    if i != j:
                        true_partners[i].add(j)
    true_partners = {k: list(v) for k, v in true_partners.items()}

    # 7. Decode and calculate metrics
    decoded_partners = decode_partners(memory_store.get_memory(), binder.E_id, top_k=5)
    recall_1 = recall_at_k(true_partners, decoded_partners, k=1)
    recall_5 = recall_at_k(true_partners, decoded_partners, k=5)

    return {
        'recall_at_1': recall_1,
        'recall_at_5': recall_5,
        'final_memory': memory_store.get_memory(),
        'hop_log': hop_log,
        'group_history': np.array(group_history),
        'true_partners': true_partners,
        'decoded_partners': decoded_partners,
    }

def main():
    parser = argparse.ArgumentParser(description="Run Fluid-HD experiments.")
    parser.add_argument('--N', type=int, default=50, help='Number of agents.')
    parser.add_argument('--T', type=int, default=1000, help='Number of simulation steps.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--output_dir', type=str, default='results/exp1', help='Directory to save results.')
    
    # Hyperparameter lists
    parser.add_argument('--D_list', nargs='+', type=int, default=[1000, 2000, 4000], help='List of HD dimensions.')
    parser.add_argument('--tau_intra_list', nargs='+', type=float, default=[1.2], help='List of intra-group affinity multipliers.')
    parser.add_argument('--tau_inter_list', nargs='+', type=float, default=[0.8], help='List of inter-group affinity multipliers.')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create hyperparameter combinations
    param_grid = list(itertools.product(args.D_list, args.tau_intra_list, args.tau_inter_list))
    
    metrics_summary = []

    for D, tau_intra, tau_inter in param_grid:
        print(f"Running with: D={D}, tau_intra={tau_intra}, tau_inter={tau_inter}")
        
        results = run_simulation(args.N, args.T, D, tau_intra, tau_inter, args.seed)
        
        run_id = f"N{args.N}_D{D}_T{args.T}_seed{args.seed}_intra{tau_intra}_inter{tau_inter}"
        
        # Save metrics
        metrics = {
            'D': D,
            'tau_intra': tau_intra,
            'tau_inter': tau_inter,
            'recall_at_1': results['recall_at_1'],
            'recall_at_5': results['recall_at_5']
        }
        metrics_summary.append(metrics)
        
        # Save raw results
        raw_output_path = os.path.join(args.output_dir, f"{run_id}_results.pkl")
        with open(raw_output_path, 'wb') as f:
            pickle.dump({
                'final_memory': results['final_memory'],
                'hop_log': results['hop_log'],
                'group_history': results['group_history'],
                'true_partners': results['true_partners'],
                'decoded_partners': results['decoded_partners'],
            }, f)

    # Save summary of metrics
    summary_path = os.path.join(args.output_dir, 'metrics_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
        
    print(f"Experiment sweep complete. Metrics saved to {summary_path}")
    # NOTE: For larger sweeps, consider using multiprocessing to run simulations in parallel.

if __name__ == '__main__':
    main()
