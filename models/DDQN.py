import random
import torch
import copy
import os
import datetime
import torch.nn as nn
import torch.optim as optim
from config.hyperpara import DDQN_param
from utils.memory import DDQN_MemoryBuffer
from utils.plot import plot_loss, plot_qvalue, plot_epsilon

class DDQN_Agent(nn.Module):
    def __init__(self, input, plot, output=2):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input, 32),
            #nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(32, 64),
            #nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(64, 128),
            #nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            #nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(64, 32),
            #nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(32, output),

        )

        self.model_target = copy.deepcopy(self.model)
        self.param = DDQN_param()
        self.optim = optim.Adam(self.model.parameters(), lr=self.param.lr)
        self.loss = nn.MSELoss()
        self.memory = DDQN_MemoryBuffer(self.param.buffer_size, self.param.batch_size)

        # counters
        self.epsilon_counter = 0 # used for updating epsilon 
        self.target_counter = 0 # used for updating target network
        self.main_counter = 0 # used for updating main network

        # plotting enable
        self.plot = plot
        if plot:
            self.loss_his = []
            self.loss_mean = []
            self.q_values = []
            self.epsilon = []
            self.plot_frequancy = self.param.plot_frequancy
            self.plot_counter = 0

        # for saving model
        self.run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.last_path_main = None
        self.last_path_target = None
        self.last_path_buffer = None

    def forward(self, x):   
        x = self.to_tensor(x)
        return self.model(x)
    
    def to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x
        else:
            return torch.tensor(x, dtype=torch.float32).unsqueeze(0) 
    
    # save the state
    def store_experience(self, curr_state, curr_action, reward, next_state, done):
        self.memory.save(curr_state, curr_action, reward, next_state, done)
    
    def get_action(self, state):
        # exploration
        if random.random() < self.param.epsilon:
            self.epsilon_counter += 1 # increse the decay counter
            force_dir = random.choice([0, 1]) # 0 apply force on the left
        # exploitation
        else:
            #self.model.eval() # needed, because of running mean and variance from batchnorm 
            output = self.forward(state)
            force_dir = torch.argmax(output) # chose the largest value
        
        # update epsilon
        if self.epsilon_counter == self.param.epislon_decay_frequancy:
            self.param.epsilon = max(self.param.epsilon_min, self.param.epsilon * self.param.epsilon_decay_rate)
            self.epsilon_counter = 0

        return force_dir

    def train_process(self):
        if self.main_counter < self.param.main_update_frequancy:
            self.main_counter += 1
            return False
        
        self.main_counter = 0 
        batch = self.memory.get_batch()
        if batch == None:
            return False
        curr_states, curr_actions, rewards, next_states, dones = zip(*batch)
        # convert it into tensors
        curr_states = torch.tensor(curr_states, dtype=torch.float) # (batch_size, state_size)
        curr_actions = torch.tensor(curr_actions, dtype=torch.int64) # (batch_size, ) ; int, because of gather()
        rewards = torch.tensor(rewards, dtype=torch.float) # (batch_size, )
        next_states = torch.tensor(next_states, dtype=torch.float) # (batch_size, state_size)
        dones = torch.tensor(dones, dtype=torch.float) # (batch_size, )

        # get Q values for the current state
        output = self.model(curr_states) # (batch_size, 2)

        # select Q values for the corresponding action
        curr_actions = curr_actions.unsqueeze(1) # (batch_size, 1)
        Q_values = output.gather(1, curr_actions).squeeze(1) # (batch_size, )

        # compute target Q value (target neural network). It does not need gradients, since this network is only for reference
        with torch.no_grad():
            # Use main model to select best actions
            output_next = self.model(next_states) # (batch_size, 2)
            best_actions = torch.argmax(output_next, dim=1) # find arg across each sample -> (batch, )
            best_actions = best_actions.unsqueeze(1) # (batch, 1)

            # Use target model to evaluate those actions
            target_output = self.model_target(next_states) # (batch_size, 2)
            selected_q_values = target_output.gather(1, best_actions).squeeze(1)

            # Calculate target Q-values
            target_Q_values = rewards + self.param.gamma * selected_q_values * (1 - dones) # perform element-wise mul.

        # compute loss
        loss = self.loss(Q_values, target_Q_values) 

        # zero the gradients
        self.optim.zero_grad()

        # backpropagation
        loss.backward()

        # clipping the gradient
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # perform a gradient step
        self.optim.step()

        # update the target 
        self.target_counter += 1
        if self.target_counter == self.param.target_update_frequancy:
            self.model_target.load_state_dict(self.model.state_dict())
            self.target_counter = 0

        # save plotting data 
        if self.plot:   
            self.plot_counter += 1
            if self.plot_counter == self.plot_frequancy:
                # average Q value of a batch
                Q_values = output.gather(1, curr_actions).squeeze(1)
                self.q_values.append(Q_values.mean().item())

                # loss and average loss
                self.loss_his.append(loss.item())
                self.loss_mean.append(sum(self.loss_his)/len(self.loss_his))

                # epsilon
                self.epsilon.append(self.param.epsilon)

                self.plot_counter = 0

    # plot model's values
    def plot_model(self):
        if self.plot and all(len(x)!=0 for x in [self.q_values, self.loss_his, self.loss_mean, self.epsilon]):
                # plot average Q value for a batch
                plot_qvalue(self.q_values)

                # loss and average loss
                plot_loss(self.loss_his, self.loss_mean)

                # epsilon
                plot_epsilon(self.epsilon)

    # save model and memory buffer
    def save_model(self, num_iter, avg_score):
        path = f"trained_models/DDQN_model/{self.run_time}"
        # Ensure path exists -> make necesarry folders
        os.makedirs(path, exist_ok=True)

        # Remove decimals
        avg_score = int(avg_score)
        
        # paths
        path_main = f"{path}/main_{num_iter}_{avg_score}.pth"
        path_target = f"{path}/target_{num_iter}_{avg_score}.pth"
        path_buffer = f"{path}/buffer_{num_iter}_{avg_score}.pkl"

        # remove old files
        if self.last_path_main and os.path.exists(self.last_path_main):
            os.remove(self.last_path_main)

        if self.last_path_target and os.path.exists(self.last_path_target):
            os.remove(self.last_path_target)

        if self.last_path_buffer and os.path.exists(self.last_path_buffer):
            os.remove(self.last_path_buffer)

        # save main network
        torch.save(self.model.state_dict(), path_main)
        # save target network
        torch.save(self.model_target.state_dict(), path_target)
        # save memory buffer
        self.memory.save_buffer(path_buffer)

        # saved paths
        self.last_path_main = path_main
        self.last_path_target = path_target
        self.last_path_buffer = path_buffer