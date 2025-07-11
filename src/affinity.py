import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

def compute_hd_embeddings(states: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Project continuous states into high-dimensional binary vectors.
    The projection is done using a random matrix W.

    Args:
        states (np.ndarray): The agent states array, shape (N, state_dim).
        W (np.ndarray): The random projection matrix, shape (D, state_dim).

    Returns:
        np.ndarray: The high-dimensional binary embeddings, shape (N, D).
    """
    # Project states into high-dimensional space
    projected = states @ W.T
    # Binarize using the sign function
    embeddings = np.sign(projected)
    # Ensure the output is integer type {-1, 1}
    return embeddings.astype(np.int8)

def cosine_affinity(H: np.ndarray, threshold: float = 0.5) -> csr_matrix:
    """
    Compute a sparse cosine similarity graph based on HD embeddings.
    An edge exists if the cosine similarity is above a certain threshold.

    Args:
        H (np.ndarray): The high-dimensional embeddings, shape (N, D).
        threshold (float): The similarity threshold to create an edge.

    Returns:
        scipy.sparse.csr_matrix: The sparse affinity matrix.
    """
    # Using sklearn's cosine_similarity is efficient
    affinity_matrix = cosine_similarity(H)
    
    # We don't want self-loops in the affinity graph
    np.fill_diagonal(affinity_matrix, 0)
    
    # Apply threshold to get a sparse representation
    affinity_matrix[affinity_matrix < threshold] = 0
    
    # Convert to a sparse matrix format
    sparse_affinity = csr_matrix(affinity_matrix)
    
    return sparse_affinity
