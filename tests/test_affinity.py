import numpy as np
import unittest
from src.affinity import compute_hd_embeddings, cosine_affinity

class TestAffinity(unittest.TestCase):

    def test_compute_hd_embeddings(self):
        """Test the shape and values of HD embeddings."""
        N, state_dim, D = 10, 4, 1000
        states = np.random.rand(N, state_dim)
        W = np.random.randn(D, state_dim)
        
        embeddings = compute_hd_embeddings(states, W)
        
        self.assertEqual(embeddings.shape, (N, D))
        self.assertTrue(np.all(np.isin(embeddings, [-1, 1])))

    def test_cosine_affinity(self):
        """Test cosine affinity matrix properties."""
        N, D = 5, 100
        # Create two identical vectors and three random ones
        H = np.random.choice([-1, 1], size=(N, D))
        H[1] = H[0] # agent 1 is identical to agent 0
        H[2] = -H[0] # agent 2 is perfectly opposite to agent 0

        affinity_matrix = cosine_affinity(H, threshold=0.5)
        
        # Should be a sparse matrix
        self.assertTrue(hasattr(affinity_matrix, 'tocsr'))
        
        # Check shape
        self.assertEqual(affinity_matrix.shape, (N, N))
        
        # The affinity between identical vectors should be high (close to 1)
        self.assertAlmostEqual(affinity_matrix[0, 1], 1.0, places=5)
        
        # The affinity between opposite vectors should be 0 (below threshold)
        self.assertEqual(affinity_matrix[0, 2], 0)
        
        # Diagonal should be zero
        self.assertEqual(affinity_matrix.diagonal().sum(), 0)

if __name__ == '__main__':
    unittest.main()
