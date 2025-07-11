import numpy as np
import random
import logging
from typing import List, Dict

def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def setup_logging(level=logging.INFO):
    """Configures basic logging."""
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

def recall_at_k(true_partners: Dict[int, List[int]], decoded_partners: Dict[int, List[int]], k: int) -> float:
    """
    Computes Recall@k.
    
    Args:
        true_partners: Ground truth dictionary mapping agent_i to list of partners.
        decoded_partners: Decoded dictionary mapping agent_i to list of partners.
        k: The 'k' in Recall@k.

    Returns:
        The average recall score.
    """
    total_recall = 0
    num_agents = 0
    
    for agent_id, true_list in true_partners.items():
        if agent_id in decoded_partners:
            num_agents += 1
            decoded_list = decoded_partners[agent_id][:k]
            true_set = set(true_list)
            if not true_set:
                continue
            
            hits = sum(1 for partner in decoded_list if partner in true_set)
            total_recall += hits / len(true_set)
            
    return total_recall / num_agents if num_agents > 0 else 0.0

def snr(signal_power: np.ndarray, noise_power: np.ndarray) -> np.ndarray:
    """
    Computes Signal-to-Noise Ratio (SNR) in dB.
    
    Args:
        signal_power: Array of signal power values.
        noise_power: Array of noise power values.
        
    Returns:
        SNR values in decibels.
    """
    # Avoid division by zero
    noise_power[noise_power == 0] = 1e-10
    ratio = signal_power / noise_power
    return 10 * np.log10(ratio)
