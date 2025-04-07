import random
import pickle
from collections import deque

class DDQN_MemoryBuffer:
    def __init__(self, size, batch_size):
        self.buffer = deque(maxlen=size)
        self.batch_size = batch_size

    def save(self, curr_state, curr_action, reward, next_state, done):
        self.buffer.append((curr_state, curr_action, reward, next_state, done))

    def get_batch(self):
        if len(self.buffer) < self.batch_size:
            return None
        else:
            return random.sample(self.buffer, self.batch_size)
        
    def save_buffer(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.buffer, f)
