import unittest
import numpy as np
import torch
import sys
import os

# Add the parent directory to the path so we can import the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dqn_agent import DQNAgent, DQNNetwork
from src.utils import ReplayBuffer


class TestDQNAgent(unittest.TestCase):
    """
    Test cases for the DQN agent
    """
    def setUp(self):
        """
        Set up the test environment
        """
        self.state_shape = (4, 84, 84)  # (stack_frames, height, width)
        self.n_actions = 5
        self.device = 'cpu'  # Use CPU for testing
        
        # Create agent
        self.agent = DQNAgent(
            state_shape=self.state_shape,
            n_actions=self.n_actions,
            device=self.device,
            buffer_size=1000,
            batch_size=32
        )
    
    def test_network_architecture(self):
        """
        Test the DQN network architecture
        """
        # Create a network
        network = DQNNetwork(self.state_shape, self.n_actions)
        
        # Create a dummy input
        dummy_input = torch.zeros(1, *self.state_shape)
        
        # Forward pass
        output = network(dummy_input)
        
        # Check output shape
        self.assertEqual(output.shape, (1, self.n_actions))
    
    def test_select_action(self):
        """
        Test action selection
        """
        # Create a dummy state
        state = np.zeros(self.state_shape)
        
        # Test with epsilon=0 (always greedy)
        self.agent.epsilon = 0
        action = self.agent.select_action(state)
        
        # Action should be an integer in [0, n_actions-1]
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.n_actions)
        
        # Test with epsilon=1 (always random)
        self.agent.epsilon = 1
        action = self.agent.select_action(state)
        
        # Action should be an integer in [0, n_actions-1]
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.n_actions)
    
    def test_update_epsilon(self):
        """
        Test epsilon update
        """
        # Set initial epsilon
        self.agent.epsilon = 1.0
        self.agent.epsilon_decay = 0.9
        self.agent.epsilon_end = 0.1
        
        # Update epsilon
        self.agent.update_epsilon()
        
        # Check new epsilon
        self.assertEqual(self.agent.epsilon, 0.9)
        
        # Update epsilon multiple times
        for _ in range(20):
            self.agent.update_epsilon()
        
        # Check that epsilon doesn't go below epsilon_end
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_end)
    
    def test_store_experience(self):
        """
        Test storing experiences in the replay buffer
        """
        # Create dummy experience
        state = np.zeros(self.state_shape)
        action = 0
        reward = 1.0
        next_state = np.ones(self.state_shape)
        done = False
        
        # Initial buffer size
        initial_size = len(self.agent.memory)
        
        # Store experience
        self.agent.store_experience(state, action, reward, next_state, done)
        
        # Check buffer size
        self.assertEqual(len(self.agent.memory), initial_size + 1)
    
    def test_save_load(self):
        """
        Test saving and loading the model
        """
        import tempfile
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.pth') as tmp:
            # Save the model
            self.agent.save(tmp.name)
            
            # Create a new agent
            new_agent = DQNAgent(
                state_shape=self.state_shape,
                n_actions=self.n_actions,
                device=self.device
            )
            
            # Load the model
            new_agent.load(tmp.name)
            
            # Check that the parameters are the same
            for param1, param2 in zip(self.agent.policy_net.parameters(), new_agent.policy_net.parameters()):
                self.assertTrue(torch.allclose(param1, param2))


if __name__ == '__main__':
    unittest.main()
