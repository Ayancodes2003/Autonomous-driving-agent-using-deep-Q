# Autonomous Driving Agent using Deep Q-Networks (DQN)

This project implements a reinforcement learning agent using Deep Q-Networks (DQN) for autonomous driving in a simulated environment. The agent learns to follow lanes and avoid obstacles in the CarRacing-v0 environment from OpenAI Gym.

## Project Structure

```
autonomous_driving_dqn/
├── models/                  # Saved models
├── src/
│   ├── train.py            # Training script
│   ├── dqn_agent.py        # DQN agent class
│   ├── environment.py      # Environment wrapper
│   └── utils.py            # Replay buffer and helper functions
├── tests/
│   └── test_agent.py       # Unit tests
├── inference.py            # Runs a trained model
├── requirements.txt        # Required packages
└── README.md               # This file
```

## Setup

### 1. Create a Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Training

To train the agent with default parameters:

```bash
python -m src.train
```

### Training Options

```
usage: train.py [-h] [--skip_frames SKIP_FRAMES] [--stack_frames STACK_FRAMES]
                [--learning_rate LEARNING_RATE] [--gamma GAMMA]
                [--epsilon_start EPSILON_START] [--epsilon_end EPSILON_END]
                [--epsilon_decay EPSILON_DECAY] [--buffer_size BUFFER_SIZE]
                [--batch_size BATCH_SIZE] [--target_update TARGET_UPDATE]
                [--num_episodes NUM_EPISODES] [--device DEVICE] [--render]
                [--seed SEED] [--model_dir MODEL_DIR] [--log_dir LOG_DIR]
                [--save_freq SAVE_FREQ]

Train a DQN agent for autonomous driving

optional arguments:
  -h, --help                      Show this help message and exit
  --skip_frames SKIP_FRAMES       Number of frames to skip (default: 4)
  --stack_frames STACK_FRAMES     Number of frames to stack (default: 4)
  --learning_rate LEARNING_RATE   Learning rate (default: 0.0001)
  --gamma GAMMA                   Discount factor (default: 0.99)
  --epsilon_start EPSILON_START   Starting epsilon for exploration (default: 1.0)
  --epsilon_end EPSILON_END       Minimum epsilon (default: 0.01)
  --epsilon_decay EPSILON_DECAY   Epsilon decay rate (default: 0.995)
  --buffer_size BUFFER_SIZE       Replay buffer size (default: 100000)
  --batch_size BATCH_SIZE         Batch size for training (default: 64)
  --target_update TARGET_UPDATE   Steps between target network updates (default: 1000)
  --num_episodes NUM_EPISODES     Number of episodes to train (default: 1000)
  --device DEVICE                 Device to use (default: cuda if available, else cpu)
  --render                        Render the environment during training
  --seed SEED                     Random seed (default: 42)
  --model_dir MODEL_DIR           Directory to save models (default: ../models)
  --log_dir LOG_DIR               Directory to save logs (default: ../logs)
  --save_freq SAVE_FREQ           Episodes between saving checkpoints (default: 50)
```

## Inference

To run a trained agent:

```bash
python inference.py --model_path models/best_model.pth
```

### Inference Options

```
usage: inference.py [-h] [--skip_frames SKIP_FRAMES]
                    [--stack_frames STACK_FRAMES] [--model_path MODEL_PATH]
                    [--num_episodes NUM_EPISODES] [--device DEVICE]
                    [--delay DELAY] [--record] [--output_dir OUTPUT_DIR]

Run inference with a trained DQN agent

optional arguments:
  -h, --help                      Show this help message and exit
  --skip_frames SKIP_FRAMES       Number of frames to skip (default: 4)
  --stack_frames STACK_FRAMES     Number of frames to stack (default: 4)
  --model_path MODEL_PATH         Path to the trained model (default: models/best_model.pth)
  --num_episodes NUM_EPISODES     Number of episodes to run (default: 5)
  --device DEVICE                 Device to use (default: cuda if available, else cpu)
  --delay DELAY                   Delay between steps for visualization (default: 0.01)
  --record                        Record video of the agent
  --output_dir OUTPUT_DIR         Directory to save videos (default: videos)
```

## Running Tests

To run the unit tests:

```bash
python -m unittest discover tests
```

## Implementation Details

### DQN Agent

The DQN agent implements the following features:
- Experience replay buffer
- Target network update
- ε-greedy exploration
- MSE loss and Adam optimizer

### Environment

The environment wrapper provides:
- Frame skipping for faster training
- Frame stacking for temporal information
- Discrete action space for the continuous CarRacing-v0 environment

## Results

During training, the agent's performance is logged and plotted. The best model is saved based on the average reward over the last 100 episodes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
