
class DDQN_param:
    def __init__(self):
        self.lr = 0.03
        self.epsilon = 1
        self.epsilon_decay_rate = 0.95
        self.epislon_decay_frequancy = 30

        self.buffer_size = 100_000