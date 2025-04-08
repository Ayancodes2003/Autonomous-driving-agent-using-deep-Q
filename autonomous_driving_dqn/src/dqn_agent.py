import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from .utils import ReplayBuffer

class DQNNetwork(nn.Module):
    """
    Deep Q-Network architecture
    """
    def __init__(self, input_shape, n_actions):
        """
        Initialize the DQN network
        
        Args:
            input_shape (tuple): Shape of the input state (frames, height, width)
            n_actions (int): Number of possible actions
        """
        super(DQNNetwork, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate the size of the output from the conv layers
        conv_output_size = self._get_conv_output_size(input_shape)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_output_size(self, input_shape):
        """
        Calculate the output size of the convolutional layers
        
        Args:
            input_shape (tuple): Shape of the input state
            
        Returns:
            int: Size of the flattened output from the conv layers
        """
        # Create a dummy input
        dummy_input = torch.zeros(1, *input_shape)
        # Forward pass through the conv layers
        conv_output = self.conv_layers(dummy_input)
        # Return the flattened size
        return int(np.prod(conv_output.shape))
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input state
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        # Forward pass through conv layers
        conv_out = self.conv_layers(x)
        # Flatten the output
        flattened = conv_out.view(conv_out.size(0), -1)
        # Forward pass through fc layers
        return self.fc_layers(flattened)


class DQNAgent:
    """
    Deep Q-Network agent
    """
    def __init__(self, state_shape, n_actions, device='cuda' if torch.cuda.is_available() else 'cpu',
                 lr=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=100000, batch_size=64, target_update=1000):
        """
        Initialize the DQN agent
        
        Args:
            state_shape (tuple): Shape of the state (frames, height, width)
            n_actions (int): Number of possible actions
            device (str): Device to use for tensor operations
            lr (float): Learning rate
            gamma (float): Discount factor
            epsilon_start (float): Starting value of epsilon for ε-greedy policy
            epsilon_end (float): Minimum value of epsilon
            epsilon_decay (float): Decay rate of epsilon
            buffer_size (int): Size of the replay buffer
            batch_size (int): Size of the batch for learning
            target_update (int): Number of steps between target network updates
        """
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Initialize networks
        self.policy_net = DQNNetwork(state_shape, n_actions).to(device)
        self.target_net = DQNNetwork(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Initialize step counter
        self.steps_done = 0
    
    def select_action(self, state, training=True):
        """
        Select an action using ε-greedy policy
        
        Args:
            state (numpy.ndarray): Current state
            training (bool): Whether the agent is in training mode
            
        Returns:
            int: Selected action index
        """
        if training and random.random() < self.epsilon:
            # Random action
            return random.randrange(self.n_actions)
        else:
            # Greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
    
    def update_epsilon(self):
        """
        Update epsilon value for ε-greedy policy
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def store_experience(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer
        
        Args:
            state (numpy.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (numpy.ndarray): Next state
            done (bool): Whether the episode has ended
        """
        self.memory.add(state, action, reward, next_state, done)
    
    def learn(self):
        """
        Update the policy network using a batch of experiences
        
        Returns:
            float: Loss value
        """
        # Check if we have enough samples in the buffer
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Move tensors to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute V(s_{t+1}) for all next states
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
        
        # Compute the expected Q values
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), expected_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to avoid exploding gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def save(self, path):
        """
        Save the model
        
        Args:
            path (str): Path to save the model
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path)
    
    def load(self, path):
        """
        Load the model
        
        Args:
            path (str): Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
