import random
import torch.nn as nn
import torch.optim as optim
from hyperpara import DDQN_param
from memory import MemoryBuffer

# need to put it to separate folder
MAX_FROCE = 10

class DDQN_Agent:
    def __init__(self, input, output=1):
        super.__init__()

        self.model = nn.Sequential(
            nn.Linear(input, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, output)
        )
        self.param = DDQN_param()
        
        self.optim = optim.Adam(self.parameters(), lr=self.param.lr)
        self.loss = nn.MSELoss()

        self.memory = MemoryBuffer(self.param.buffer_size)

        # counters
        self.epsilon_counter = 0

    def forward(self, x):
        return self.model(x)
    
    def get_action(self, state):
        # exploration
        if random.random() < self.param.epsilon:
            self.epsilon_counter += 1 # increse the decay counter
            force = random.uniform(-MAX_FROCE, MAX_FROCE)
        # exploitation
        else:
            self.model.eval()
            force = max(-MAX_FROCE, min(self.forward(state), MAX_FROCE)) # clip the force within the valid range
        
        # update epsilon
        if self.epsilon_counter == self.param.epislon_decay_frequancy:
            self.param.epsilon = self.param.epsilon * self.param.epsilon_decay_rate

        return force

    def train_process(self):
        pass
