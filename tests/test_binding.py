import unittest
import numpy as np
from src.binding import HDBinder
from src.memory import MemoryStore

class TestBinding(unittest.TestCase):

    def setUp(self):
        """Set up a binder and memory store for tests."""
        self.N, self.D, self.seed = 4, 128, 42
        self.binder = HDBinder(self.N, self.D, self.seed)
        self.memory_store = MemoryStore(self.D)

    def test_id_vector_creation(self):
        """Test the properties of the created ID vectors."""
        self.assertEqual(self.binder.E_id.shape, (self.N, self.D))
        self.assertTrue(np.all(np.isin(self.binder.E_id, [-1, 1])))

    def test_bind_operation(self):
        """Test the binding (element-wise product) operation."""
        v1 = self.binder.E_id[0]
        v2 = self.binder.E_id[1]
        bound = self.binder.bind(0, 1)
        
        self.assertTrue(np.array_equal(bound, v1 * v2))
        # Binding with self should result in a vector of all ones
        self.assertTrue(np.all(self.binder.bind(0, 0) == 1))

    def test_record_co_occurrence(self):
        """Test if co-occurrence is correctly bundled into memory."""
        group = [0, 1, 2]
        self.binder.record_co_occurrence(self.memory_store, group)
        
        # Expected memory is the sum of bound pairs: (0,1), (0,2), (1,2)
        m_expected = (self.binder.bind(0, 1) + 
                      self.binder.bind(0, 2) + 
                      self.binder.bind(1, 2))
        
        self.assertTrue(np.array_equal(self.memory_store.get_memory(), m_expected))
        
        # Test with a group of less than 2
        self.memory_store.reset()
        self.binder.record_co_occurrence(self.memory_store, [3])
        self.assertTrue(np.all(self.memory_store.get_memory() == 0))


if __name__ == '__main__':
    unittest.main()
