import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.binding import HDBinder

class MemoryStore:
    def __init__(self, D: int):
        """
        Initializes the memory store.

        Args:
            D (int): The dimensionality of the memory vector.
        """
        self.D = D
        self.M = np.zeros(D, dtype=np.int32)

    def get_memory(self) -> np.ndarray:
        """Returns a copy of the memory vector."""
        return self.M.copy()

    def add(self, vector: np.ndarray):
        """Adds a vector to the memory."""
        self.M += vector

    def normalize(self):
        """
        Normalizes the memory vector by taking the sign of its components.
        This is a form of hard quantization.
        """
        self.M = np.sign(self.M).astype(np.int8)

    def reset(self):
        """Resets the memory to zeros."""
        self.M = np.zeros(self.D, dtype=np.int32)
