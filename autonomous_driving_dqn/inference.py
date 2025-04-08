import os
import argparse
import torch
import numpy as np
import time
import cv2
from src.environment import CarRacingEnv
from src.dqn_agent import DQNAgent

def run_inference(args):
    """
    Run inference with a trained DQN agent
    
    Args:
        args: Command line arguments
    """
    # Create environment
    env = CarRacingEnv(skip_frames=args.skip_frames, stack_frames=args.stack_frames)
    
    # Get state shape and number of actions
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    # Create agent
    agent = DQNAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        device=args.device
    )
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    agent.load(args.model_path)
    print("Model loaded successfully!")
    
    # Set up video recording if specified
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(args.output_dir, f'inference_{int(time.time())}.mp4')
        video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (600, 400))
        print(f"Recording video to {video_path}")
    
    # Run episodes
    for episode in range(args.num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        print(f"Starting episode {episode+1}/{args.num_episodes}")
        
        while not done:
            # Select action (no exploration during inference)
            action = agent.select_action(state, training=False)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            step += 1
            
            # Render
            env.render()
            
            # Record video if specified
            if args.record:
                # Get the rendered frame
                frame = env.env.render(mode='rgb_array')
                # Write to video
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            # Delay for better visualization
            time.sleep(args.delay)
            
            # Print step information
            if step % 10 == 0:
                print(f"Step {step} | Action: {action} | Reward: {reward:.2f} | Total Reward: {episode_reward:.2f}")
        
        print(f"Episode {episode+1} finished with total reward: {episode_reward:.2f}")
    
    # Close video writer if recording
    if args.record:
        video_writer.release()
        print(f"Video saved to {video_path}")
    
    # Close environment
    env.close()


def main():
    """
    Main function to parse arguments and start inference
    """
    parser = argparse.ArgumentParser(description='Run inference with a trained DQN agent')
    
    # Environment parameters
    parser.add_argument('--skip_frames', type=int, default=4, help='Number of frames to skip')
    parser.add_argument('--stack_frames', type=int, default=4, help='Number of frames to stack')
    
    # Inference parameters
    parser.add_argument('--model_path', type=str, default='models/best_model.pth', help='Path to the trained model')
    parser.add_argument('--num_episodes', type=int, default=5, help='Number of episodes to run')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--delay', type=float, default=0.01, help='Delay between steps for visualization')
    
    # Recording parameters
    parser.add_argument('--record', action='store_true', help='Record video of the agent')
    parser.add_argument('--output_dir', type=str, default='videos', help='Directory to save videos')
    
    args = parser.parse_args()
    
    # Create output directory if recording
    if args.record:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Run inference
    run_inference(args)


if __name__ == '__main__':
    main()
