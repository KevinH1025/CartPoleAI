import random
import math
import settings

# constants
g = 9.81    # gravity 
M = 1       # cart's mass
m = 0.5     # pole's mass
l = 0.5     # pole's length
dt = 0.02   # simulation time step

# optional settings
c = 0       # friction between cart and ground
p = 0       # friction between pole and cart

# limits before restart
pos_lim = settings.WIDTH                 # max position
angle_lim = math.radians(30)    # max angle difference

class CartPole:
    def __init__(self):
        self.pos = random.uniform(0, pos_lim) # cart's initial postion
        self.velocity = 0 # cart's initial velocity
        self.angle = random.uniform(-angle_lim, angle_lim) # pole's initial angle
        self.angular_velocity = 0 # pole's initial angular velocity
        self.force = 0 # initial force applied to the cart

    # one simulation step
    def move(self):
        temp = (self.force + m * l * self.angle**2 * math.sin(self.angle)) / (M + m)
        # pole angular acceleration
        angular_acc = (g * math.sin(self.angle) - math.cos(self.angle) * temp) / (l * (4/3 - m * math.cos(self.angle)**2) / (M + m))
        # cart acceleration
        cart_acc = temp - m * l * angular_acc * math.cos(self.angle) / (M + m)

        # update the values
        self.velocity += cart_acc * dt
        self.angular_velocity += angular_acc * dt

        self.pos += self.velocity * dt
        self.angle += self.angular_velocity * dt

        # check if exceeded the limit
        self.reset()

    # end the episode if exceed the limit
    def reset(self):
        if self.pos < 0 or self.pos > pos_lim or self.angle < -angle_lim or self.angle > angle_lim:
            self.pos = random.uniform(0, pos_lim) 
            self.velocity = 0 
            self.angle = random.uniform(-angle_lim, angle_lim) 
            self.angular_velocity = 0 
    
    # get the current state
    def get_state(self):
        return self.pos, self.velocity, self.angle, self.angular_velocity