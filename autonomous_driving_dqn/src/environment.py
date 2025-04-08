import gym
import numpy as np
from gym.spaces import Discrete
from .utils import preprocess_state

class CarRacingEnv:
    """
    Wrapper for the CarRacing-v0 environment from OpenAI Gym
    """
    def __init__(self, skip_frames=4, stack_frames=4, seed=None):
        """
        Initialize the environment wrapper
        
        Args:
            skip_frames (int): Number of frames to skip between actions
            stack_frames (int): Number of frames to stack for the state
            seed (int, optional): Random seed for reproducibility
        """
        self.env = gym.make('CarRacing-v0', render_mode='human')
        if seed is not None:
            self.env.seed(seed)
            np.random.seed(seed)
        
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames
        
        # Define discrete actions
        # 0: Straight, 1: Left, 2: Right, 3: Accelerate, 4: Brake
        self.actions = [
            [0, 0, 0],    # No action (straight)
            [-1, 0, 0],   # Left
            [1, 0, 0],    # Right
            [0, 1, 0],    # Accelerate
            [0, 0, 0.8]   # Brake
        ]
        
        # Define action space
        self.action_space = Discrete(len(self.actions))
        
        # Define observation space (stacked frames)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, 
            shape=(self.stack_frames, 84, 84), 
            dtype=np.float32
        )
        
        # Frame stack
        self.frame_stack = None
    
    def reset(self):
        """
        Reset the environment and return the initial state
        
        Returns:
            numpy.ndarray: Initial state as stacked frames
        """
        initial_state = self.env.reset()
        processed_frame = preprocess_state(initial_state)
        
        # Initialize frame stack
        self.frame_stack = [processed_frame] * self.stack_frames
        
        return np.array(self.frame_stack)
    
    def step(self, action_idx):
        """
        Take an action in the environment
        
        Args:
            action_idx (int): Index of the action to take
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        total_reward = 0
        done = False
        info = {}
        
        # Get the actual action from the action index
        action = self.actions[action_idx]
        
        # Skip frames
        for _ in range(self.skip_frames):
            if not done:
                next_state, reward, done, info = self.env.step(action)
                total_reward += reward
        
        # Preprocess the next state
        processed_frame = preprocess_state(next_state)
        
        # Update frame stack
        self.frame_stack.pop(0)
        self.frame_stack.append(processed_frame)
        
        return np.array(self.frame_stack), total_reward, done, info
    
    def render(self):
        """
        Render the environment
        """
        self.env.render()
    
    def close(self):
        """
        Close the environment
        """
        self.env.close()
