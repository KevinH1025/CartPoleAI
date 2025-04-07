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
│   │   ├── model.pth          # Trained DDQN model
│   │   └── replay_buffer.pkl  # Replay buffer
│   └── PPO_model/             # PPO model storage (Planned) 
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
- Experience replay buffer
- Mitigation of catastrophic forgetting

----

## Set Up

### Requirements
- Python 3.8+
- Install dependencies:
```
pip install -r requirements.txt
```

### Run Training
```
python main.py
```

### Run Evaluation
```
python main.py
```

---

## TODO
- Add PPO and Dueling DDQN
- Spawn CartPole randomly, which will yield better result during evaluation
- More advanced reward shaping strategies 
- Save/load model checkpoints


