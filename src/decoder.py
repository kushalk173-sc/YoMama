import numpy as np
from typing import Dict, List
from sklearn.metrics.pairwise import cosine_similarity

def decode_partners(M: np.ndarray, E_id: np.ndarray, top_k: int) -> Dict[int, List[int]]:
    """
    Decodes interaction partners for each agent from the memory vector.

    For each agent i, it unbinds its ID vector from the memory to get a noisy
    representation of its partners' IDs. It then finds the top_k agents whose
    ID vectors are most similar to this result.

    Args:
        M (np.ndarray): The final memory vector.
        E_id (np.ndarray): The matrix of agent ID vectors, shape (N, D).
        top_k (int): The number of top partners to return for each agent.

    Returns:
        Dict[int, List[int]]: A dictionary mapping each agent's index to a list
                               of its top_k decoded partner indices.
    """
    N = E_id.shape[0]
    decoded_partners = {}

    for i in range(N):
        # Unbind agent i's ID from the memory vector
        # This is equivalent to binding, M * E_id[i]
        unbound_vector = M * E_id[i]

        # Calculate cosine similarity between the unbound vector and all agent IDs
        # The result is a noisy sum of partner vectors, so we find which E_j it's closest to
        similarities = cosine_similarity(unbound_vector.reshape(1, -1), E_id).flatten()
        
        # We should not decode the agent itself as a partner
        similarities[i] = -np.inf

        # Get the indices of the top_k most similar agents
        # Use argpartition for efficiency if k is small, otherwise argsort
        if top_k < N / 2:
            # Get the indices of the top_k elements
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            # Sort these top_k indices by similarity score
            top_partners = top_indices[np.argsort(similarities[top_indices])][::-1]
        else:
            top_partners = np.argsort(similarities)[::-1][:top_k]
        
        decoded_partners[i] = top_partners.tolist()

    return decoded_partners
