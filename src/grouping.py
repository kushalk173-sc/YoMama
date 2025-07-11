import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def build_adjacency(aff: csr_matrix, prev_groups: np.ndarray, tau_intra: float, tau_inter: float) -> csr_matrix:
    """
    Builds an adjacency matrix by modulating affinity based on previous group memberships.
    This introduces hysteresis, making groups more stable.

    Args:
        aff (csr_matrix): The base affinity matrix (from cosine similarity).
        prev_groups (np.ndarray): Array of group labels from the previous time step.
        tau_intra (float): Multiplier for affinities within the same group (should be > 1).
        tau_inter (float): Multiplier for affinities between different groups (should be < 1).

    Returns:
        csr_matrix: The modulated adjacency matrix.
    """
    A_mod = aff.copy().tolil()

    # Get the coordinates of non-zero elements in the affinity matrix
    rows, cols = A_mod.nonzero()

    for i, j in zip(rows, cols):
        # Apply modulation only on the upper triangle to avoid double work
        if i >= j:
            continue
        
        if prev_groups[i] == prev_groups[j]:
            # Agents were in the same group, strengthen the link
            A_mod[i, j] *= tau_intra
            A_mod[j, i] = A_mod[i, j] # Maintain symmetry
        else:
            # Agents were in different groups, weaken the link
            A_mod[i, j] *= tau_inter
            A_mod[j, i] = A_mod[i, j] # Maintain symmetry
    
    return A_mod.tocsr()

def extract_groups(A: csr_matrix) -> np.ndarray:
    """
    Extracts group labels from an adjacency matrix using connected components.

    Args:
        A (csr_matrix): The adjacency matrix.

    Returns:
        np.ndarray: An array where the i-th element is the group ID of agent i.
    """
    n_components, labels = connected_components(csgraph=A, directed=False, return_labels=True)
    return labels
