import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import logging
from datetime import datetime

from .environment import CarRacingEnv
from .dqn_agent import DQNAgent
from .utils import plot_rewards, create_directory, get_current_time_str

def train(args):
    """
    Train the DQN agent
    
    Args:
        args: Command line arguments
    """
    # Set up logging
    log_dir = os.path.join(args.log_dir, get_current_time_str())
    create_directory(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    # Set random seeds for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Create environment
    env = CarRacingEnv(skip_frames=args.skip_frames, stack_frames=args.stack_frames, seed=args.seed)
    
    # Get state shape and number of actions
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    # Create agent
    agent = DQNAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        device=args.device,
        lr=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update=args.target_update
    )
    
    # Training metrics
    rewards = []
    avg_rewards = []
    best_avg_reward = -float('inf')
    
    # Training loop
    logging.info(f"Starting training for {args.num_episodes} episodes...")
    
    for episode in range(args.num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        # Episode loop
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)
            
            # Learn
            loss = agent.learn()
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Render if specified
            if args.render:
                env.render()
        
        # Update epsilon
        agent.update_epsilon()
        
        # Record metrics
        rewards.append(episode_reward)
        avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        avg_rewards.append(avg_reward)
        
        # Log progress
        logging.info(f"Episode {episode+1}/{args.num_episodes} | Reward: {episode_reward:.2f} | Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.4f}")
        
        # Save best model
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            model_path = os.path.join(args.model_dir, 'best_model.pth')
            agent.save(model_path)
            logging.info(f"New best model saved with average reward: {best_avg_reward:.2f}")
        
        # Save checkpoint
        if (episode + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.model_dir, f'checkpoint_episode_{episode+1}.pth')
            agent.save(checkpoint_path)
            logging.info(f"Checkpoint saved at episode {episode+1}")
            
            # Plot and save rewards
            plot_path = os.path.join(log_dir, f'rewards_episode_{episode+1}.png')
            plot_rewards(rewards, avg_rewards, filename=plot_path)
    
    # Save final model
    final_model_path = os.path.join(args.model_dir, 'final_model.pth')
    agent.save(final_model_path)
    logging.info(f"Final model saved after {args.num_episodes} episodes")
    
    # Plot final rewards
    final_plot_path = os.path.join(log_dir, 'final_rewards.png')
    plot_rewards(rewards, avg_rewards, filename=final_plot_path)
    
    # Close environment
    env.close()
    
    return rewards, avg_rewards


def main():
    """
    Main function to parse arguments and start training
    """
    parser = argparse.ArgumentParser(description='Train a DQN agent for autonomous driving')
    
    # Environment parameters
    parser.add_argument('--skip_frames', type=int, default=4, help='Number of frames to skip')
    parser.add_argument('--stack_frames', type=int, default=4, help='Number of frames to stack')
    
    # Agent parameters
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='Starting epsilon for exploration')
    parser.add_argument('--epsilon_end', type=float, default=0.01, help='Minimum epsilon')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--buffer_size', type=int, default=100000, help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--target_update', type=int, default=1000, help='Steps between target network updates')
    
    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--render', action='store_true', help='Render the environment during training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Saving parameters
    parser.add_argument('--model_dir', type=str, default='../models', help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='../logs', help='Directory to save logs')
    parser.add_argument('--save_freq', type=int, default=50, help='Episodes between saving checkpoints')
    
    args = parser.parse_args()
    
    # Create directories
    create_directory(args.model_dir)
    create_directory(args.log_dir)
    
    # Start training
    train(args)


if __name__ == '__main__':
    main()
