import numpy as np
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.memory import MemoryStore

class HDBinder:
    def __init__(self, N: int, D: int, seed: int):
        """
        Initializes the Hyperdimensional Binder.

        Args:
            N (int): The number of agents.
            D (int): The dimensionality of the hypervectors.
            seed (int): Random seed for reproducibility.
        """
        self.N = N
        self.D = D
        self.seed = seed
        np.random.seed(self.seed)
        
        # Create random bipolar ID vectors for each agent
        self.E_id = self._create_id_vectors()

    def _create_id_vectors(self) -> np.ndarray:
        """Generates N random bipolar vectors of dimensionality D."""
        return np.random.choice([-1, 1], size=(self.N, self.D), p=[0.5, 0.5])

    def bind(self, i: int, j: int) -> np.ndarray:
        """
        Binds the ID vectors of agent i and agent j.
        Binding is done by element-wise multiplication (XOR).

        Args:
            i (int): Index of the first agent.
            j (int): Index of the second agent.

        Returns:
            np.ndarray: The bound vector representing the pair (i, j).
        """
        if i >= self.N or j >= self.N:
            raise ValueError("Agent index out of bounds.")
        
        return self.E_id[i] * self.E_id[j]

    def record_co_occurrence(self, memory_store: 'MemoryStore', group: List[int]):
        """
        Records all pairwise co-occurrences within a group and bundles them into memory.
        Uses an efficient method to compute the sum of all pairwise bindings.
        
        Args:
            memory_store (MemoryStore): The memory store to update.
            group (List[int]): A list of agent indices in the same group.
        """
        if len(group) < 2:
            return

        # Sum the ID vectors of all agents in the group
        group_vectors = self.E_id[group]
        S = np.sum(group_vectors, axis=0)

        # The sum of all pairwise products is (S^2 - n*1) / 2
        # where S^2 is element-wise square. For bipolar vectors, S^2 is not simple.
        # Let's stick to the simple, correct, albeit slower version for now to avoid errors.
        # The optimization is non-trivial with integer vectors.
        # The prompt `M += e_id[i] * e_id[j]` is what is implemented here.
        
        bundle = np.zeros_like(self.E_id[0])
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                agent1_idx = group[i]
                agent2_idx = group[j]
                
                bound_pair = self.bind(agent1_idx, agent2_idx)
                bundle += bound_pair
        
        memory_store.add(bundle)
