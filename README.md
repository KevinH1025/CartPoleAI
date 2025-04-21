# CartPoleAI

A reinforcement learning project to balance a pole on a moving cart using various algorithms. 
This project uses custom environment and reward shaping. 
It aims to implement and compare different RL methods like Double Deep Q-Networks (DDQN), Proximity Policy Optimization, and more.

---

## Structure
```
CartPoleAI/
├── config/                  # Configuration files
│   ├── constants.py         # Constants for the environment and game logic
│   ├── display.py           # Display and rendering settings
│   └── hyperpara.py         # Training hyperparameters (learning rate, gamma, etc.)
│
├── models/                  # Neural network architectures
│   ├── DDQN.py              # Deep Double Q-Network
│   └── PPO.py               # PPO model (Planned)
│
├── trained_models/          # Saved models and buffers
│   ├── DDQN_model/          
│   │   └── Perfect_model/     # Can balance > 4 min
│   │       ├── buffer.pkl     # Replay buffer
│   │       ├── main.pth       # Main network 
│   │       └── target.pth     # Target network
│   └── PPO_model/           # PPO model storage (Planned) 
│
├── utils/                   
│   ├── memory.py            # Replay memory buffer
│   └── UI.py                # Custom UI functions (rendering, font handling)
│
├── cartPole.py              # Custom CartPole environment
├── main.py                  # Entry point for training/evaluation
├── requirements.txt         # Project dependencies
└── Arial.ttf                # Font file used in UI rendering
```

---

## Features

- Custom CartPole environment using Pygame (not OpenAI Gym)
- Reward shaping for more stable learning
- Deep Double Q-Networks (DDQN)
- Experience replay buffer -> Mitigation of catastrophic forgetting
- Save/load model checkpoints

---

## Set Up

### Requirements
- Python 3.8+
- Install dependencies:

```bash
pip install -r requirements.txt
```

### Arguments:

This project uses command-line arguments to control training and testing.

- `--mode`: `train` to train an agent, or `test` to test a trained model
- `--algo`: `ddqn` or `ppo` (currently only `ddqn` is implemented)
- `--render`: (optional) Enables Pygame rendering during training. Automatically enabled during testing
- `--model_path`: Path to the trained model folder (required for test mode)

### Examples

**Train a DDQN agent (with rendering):**

```bash
python3 main.py --mode train --algo ddqn --render
```

**Train a DDQN agent (no rendering):**

```bash
python3 main.py --mode train --algo ddqn
```

**Test a trained DDQN agent (always renders):**

```
python3 main.py --mode test --algo ddqn --model_path trained_models/DDQN_model/choose_your_model
```

### Pretraied Model

Under `trained_models/DDQN_model/Perfect_model/` you’ll find a pretrained network that can balance the pole for over 4 minutes. To test it, run:

```bash
python3 main.py --mode test --algo ddqn --model_path trained_models/DDQN_model/Perfect_model
```

A Pygame window will pop up with the CartPole spawning in the center of the screen.

---

### TensorBoard Visualization

This project logs real-time training metrics for visualization and performance monitoring.

#### View in VSCode

1. Install the **TensorBoard** extension
2. Start training, logs will be saved in `runs/`
3. Open Command Palette (`Cmd+Shift+P` on Mac or `Ctrl+Shift+P` on Windows)
4. Select `Python: Launch TensorBoard`,
5. Choose `Select another folder`, then navigate to a specific run inside:
   ```
   CartPoleAI/runs/<current_run>/
   ```
7. TensorBoard will open in a new VSCode tab and visualize metrics for the currrent run

#### View in browser

While training is running, open new terminal and run:

```bash
python3 -m tensorboard.main --logdir runs
```

Then visit:

```
http://localhost:6006
```

---

## TODO
- Implement Proximal Policy Optimization (PPO) and Dueling DDQN  
- Add an option to resume training from saved checkpoints
- Include a small animated GIF under **Pretrained Model** showcasing the CartPole balancing process
