import random
import math
from settings import WIDTH

# constants
G = 9.81    # gravity 
M = 1       # cart's mass
m = 0.5     # pole's mass
l = 0.5     # pole's length

# optional settings
c = 0       # friction between cart and ground
p = 0       # friction between pole and cart

# limits before restart
pos_lim = WIDTH                 # max position
angle_lim = math.radians(30)    # max angle difference

class CartPole:
    def __init__(self):
        self.pos = random.uniform(0, pos_lim) # cart's postion
        self.angle = random.uniform(-angle_lim, angle_lim) # pole angle
        self.velocity = 0 # cart's velocity

    def move(self):
        pass

    def reset(self):
        pass

    def get_state(self):
        return 