
class DDQN_param:
    def __init__(self):
        self.lr = 0.001
        self.gamma = 0.99
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay_rate = 0.95
        self.epislon_decay_frequancy = 50

        self.main_update_frequancy = 0
        self.target_update_frequancy = 500

        self.buffer_size = 100_000
        self.batch_size = 1024

        self.plot_frequancy = 10