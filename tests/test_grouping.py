import unittest
import numpy as np
from scipy.sparse import csr_matrix
from src.grouping import build_adjacency, extract_groups

class TestGrouping(unittest.TestCase):

    def test_extract_groups(self):
        """Test group extraction from a simple adjacency matrix."""
        # A matrix representing two distinct groups: {0, 1} and {2, 3}
        A = csr_matrix(np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ]))
        
        groups = extract_groups(A)
        self.assertEqual(groups.shape[0], 4)
        # The first two should be in the same group, and the last two in another
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[2], groups[3])
        self.assertNotEqual(groups[0], groups[2])

    def test_build_adjacency(self):
        """Test adjacency matrix modulation."""
        # Base affinity: all connected
        aff = csr_matrix(np.ones((4, 4)) - np.eye(4))
        # Previous groups: {0, 1} and {2, 3}
        prev_groups = np.array([0, 0, 1, 1])
        tau_intra = 1.5
        tau_inter = 0.5
        
        A_mod = build_adjacency(aff, prev_groups, tau_intra, tau_inter)
        
        # Check intra-group affinity modulation
        self.assertAlmostEqual(A_mod[0, 1], 1.0 * tau_intra)
        self.assertAlmostEqual(A_mod[2, 3], 1.0 * tau_intra)
        
        # Check inter-group affinity modulation
        self.assertAlmostEqual(A_mod[0, 2], 1.0 * tau_inter)
        self.assertAlmostEqual(A_mod[1, 3], 1.0 * tau_inter)

if __name__ == '__main__':
    unittest.main()
