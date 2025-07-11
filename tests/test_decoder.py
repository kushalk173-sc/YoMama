import unittest
import numpy as np
from src.decoder import decode_partners
from src.binding import HDBinder

class TestDecoder(unittest.TestCase):

    def setUp(self):
        """Set up a binder and memory for testing."""
        self.N, self.D, self.seed = 5, 1024, 42
        self.binder = HDBinder(self.N, self.D, self.seed)
    
    def test_decode_partners_simple(self):
        """Test partner decoding with a simple, noise-free memory."""
        # Create a memory representing the pair (0, 1) and (0, 2)
        M = self.binder.bind(0, 1) + self.binder.bind(0, 2)
        
        # Decode partners for agent 0
        decoded = decode_partners(M, self.binder.E_id, top_k=2)
        
        # The top 2 partners for agent 0 should be 1 and 2
        self.assertIn(1, decoded[0])
        self.assertIn(2, decoded[0])
        self.assertEqual(len(decoded[0]), 2)
        
        # Agent 1's top partner should be 0
        decoded_1 = decode_partners(M, self.binder.E_id, top_k=1)
        self.assertEqual(decoded_1[1][0], 0)

    def test_decode_partners_with_noise(self):
        """Test partner decoding with a slightly noisy memory."""
        # Memory with pairs (0,1)x5, (0,2)x3, and a noise pair (3,4)
        M = (5 * self.binder.bind(0, 1) + 
             3 * self.binder.bind(0, 2) +
             1 * self.binder.bind(3, 4))
             
        # Normalize to simulate a more realistic memory
        M = np.sign(M)
        
        # Decode for agent 0
        decoded = decode_partners(M, self.binder.E_id, top_k=2)
        
        # The top partner for agent 0 should be 1, followed by 2
        # due to the higher count.
        self.assertEqual(decoded[0][0], 1)
        self.assertEqual(decoded[0][1], 2)
        
if __name__ == '__main__':
    unittest.main()
