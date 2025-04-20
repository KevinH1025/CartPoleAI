import math
from config import display

# constants
g = 9.81    # gravity 
M = 1       # cart's mass
F = 10      # force applied at each step (either from left or right) 
m = 0.1     # pole's mass
l = 0.5     # pole's length
dt = 0.02   # simulation step time

# optional settings
c = 0       # friction between cart and ground
p = 0       # friction between pole and cart

# ----- environment's limit ------
# range for inital pole's angle
init_angle = math.radians(15)

# limits before episode end
pos_lim = (display.WIDTH-display.CART_W)/display.scaling    # max position -> do not change it is based on the size of window
angle_lim = math.radians(30)                                # max angle difference

# what score does the agent has to reach (in sec)
max_score = 4 * 60 