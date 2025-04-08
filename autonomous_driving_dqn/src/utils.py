import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import os
import time

# Define a named tuple for storing experiences
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """
    Replay Buffer for storing and sampling experiences
    """
    def __init__(self, capacity):
        """
        Initialize a ReplayBuffer object.
        
        Args:
            capacity (int): Maximum size of the buffer
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer
        
        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether the episode has ended
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer
        
        Args:
            batch_size (int): Size of the batch to sample
            
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones)
        """
        experiences = random.sample(self.buffer, k=batch_size)
        
        # Convert to tensors
        states = torch.from_numpy(np.array([e.state for e in experiences])).float()
        actions = torch.tensor([e.action for e in experiences]).long()
        rewards = torch.tensor([e.reward for e in experiences]).float()
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences])).float()
        dones = torch.tensor([e.done for e in experiences]).float()
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return the current size of the buffer"""
        return len(self.buffer)


def preprocess_state(state):
    """
    Preprocess the state image from the environment
    
    Args:
        state (numpy.ndarray): RGB image from the environment
        
    Returns:
        numpy.ndarray: Preprocessed grayscale image
    """
    import cv2
    # Convert to grayscale
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    # Resize to 84x84
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    # Normalize
    normalized = resized / 255.0
    return normalized


def plot_rewards(rewards, avg_rewards, filename=None):
    """
    Plot the rewards and average rewards over episodes
    
    Args:
        rewards (list): List of rewards for each episode
        avg_rewards (list): List of average rewards over episodes
        filename (str, optional): If provided, save the plot to this file
    """
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Rewards')
    plt.plot(avg_rewards, label='Average Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    
    if filename:
        plt.savefig(filename)
    
    plt.show()


def create_directory(directory):
    """
    Create a directory if it doesn't exist
    
    Args:
        directory (str): Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_current_time_str():
    """
    Get current time as a string for logging purposes
    
    Returns:
        str: Current time in format YYYYMMDD_HHMMSS
    """
    return time.strftime("%Y%m%d_%H%M%S")
