import numpy as np

class AgentSimulator:
    def __init__(self, N: int, D_space: int = 2, bar_positions=None, density_radius: float = 0.1, speed_mean: float = 0.01, speed_std: float = 0.005, goal_drift_strength: float = 0.01, jitter_strength: float = 0.001, seed: int = 42):
        """
        Initialize the agent simulator.

        Args:
            N (int): Number of agents.
            D_space (int): Dimensionality of the space (default is 2 for (x, y)).
            bar_positions (np.ndarray, optional): Positions of "bars" or attractors. Defaults to a single bar at the center.
            density_radius (float): Radius to check for overcrowding.
            speed_mean (float): Mean of the initial agent speeds.
            speed_std (float): Standard deviation of the initial agent speeds.
            goal_drift_strength (float): How strongly agents drift towards goals (bars).
            jitter_strength (float): Magnitude of random jitter in agent movement.
            seed (int): Random seed for reproducibility.
        """
        self.N = N
        self.D_space = D_space
        self.seed = seed
        np.random.seed(self.seed)

        # State: [x, y, vx, vy] for each agent
        self.state = np.zeros((N, D_space * 2))
        self.state[:, :D_space] = np.random.rand(N, D_space) # Initial positions
        
        # Initial velocities
        velocities = np.random.normal(speed_mean, speed_std, (N, D_space))
        self.state[:, D_space:] = velocities
        
        if bar_positions is None:
            self.bar_positions = np.array([[0.5, 0.5]])
        else:
            self.bar_positions = bar_positions
            
        self.density_radius = density_radius
        self.goal_drift_strength = goal_drift_strength
        self.jitter_strength = jitter_strength

    def _apply_overcrowding(self):
        """Agents move away from dense areas."""
        for i in range(self.N):
            pos_i = self.state[i, :self.D_space]
            # Find neighbors within the density radius
            distances = np.linalg.norm(self.state[:, :self.D_space] - pos_i, axis=1)
            neighbors = np.where((distances < self.density_radius) & (distances > 0))[0]
            
            if len(neighbors) > 0:
                # Calculate repulsion vector
                repulsion_vec = np.sum(pos_i - self.state[neighbors, :self.D_space], axis=0)
                repulsion_vec /= (np.linalg.norm(repulsion_vec) + 1e-6) # Normalize
                # Apply a small push away from the crowd center
                self.state[i, D_space:] += repulsion_vec * self.jitter_strength * 2

    def _apply_goal_drift(self):
        """Agents drift towards the nearest bar."""
        for i in range(self.N):
            pos_i = self.state[i, :self.D_space]
            # Find the closest bar
            bar_distances = np.linalg.norm(self.bar_positions - pos_i, axis=1)
            closest_bar = self.bar_positions[np.argmin(bar_distances)]
            # Update velocity to drift towards the bar
            drift_vec = closest_bar - pos_i
            drift_vec /= (np.linalg.norm(drift_vec) + 1e-6) # Normalize
            self.state[i, D_space:] += drift_vec * self.goal_drift_strength

    def _apply_jitter(self):
        """Apply random jitter to movement."""
        jitter = np.random.normal(0, self.jitter_strength, (self.N, self.D_space))
        self.state[:, D_space:] += jitter

    def _handle_boundaries(self, boundary_min=0.0, boundary_max=1.0):
        """Handle boundary collisions by reversing velocity."""
        for i in range(self.N):
            for j in range(self.D_space):
                if self.state[i, j] < boundary_min or self.state[i, j] > boundary_max:
                    self.state[i, j] = np.clip(self.state[i, j], boundary_min, boundary_max)
                    self.state[i, self.D_space + j] *= -1 # Reverse velocity component

    def step(self) -> np.ndarray:
        """
        Apply simulation rules for one time step.

        Returns:
            np.ndarray: The updated state array of shape (N, 4).
        """
        self._apply_overcrowding()
        self._apply_goal_drift()
        self._apply_jitter()
        
        # Update positions based on velocities
        self.state[:, :self.D_space] += self.state[:, D_space:]
        
        # Handle boundaries
        self._handle_boundaries()
        
        return self.state.copy()
