
class DDQN_param:
    def __init__(self):
        self.lr = 0.03
        self.gamma = 0.98
        self.epsilon = 1
        self.epsilon_min = 0.001
        self.epsilon_decay_rate = 0.95
        self.epislon_decay_frequancy = 30

        self.main_update_frequancy = 4
        self.target_update_frequancy = 1000

        self.buffer_size = 100_000
        self.batch_size = 1024