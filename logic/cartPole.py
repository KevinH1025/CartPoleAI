import random
import math
import logic.settings as settings
import utils.constants as constants
import pygame
from utils.plot import plot_score

# range for inital pole's angle
init_angle = math.radians(10)

# scaling factor
scaling = 100

# limits before restart
pos_lim = (settings.WIDTH-settings.CART_W)/scaling  # max position
angle_lim = math.radians(24)                        # max angle difference

class CartPole:
    def __init__(self, plot):
        self.pos = [settings.WIDTH/(2 * scaling), 2/3 * settings.HEIGHT] # cart's initial postion, in the middle of line (x-axis scaled into meters)
        self.velocity = 0 # cart's initial velocity
        self.angle = random.uniform(-init_angle, init_angle) # pole's initial angle

        self.angular_velocity = 0 # pole's initial angular velocity
        self.force = 0 # initial force applied to the cart

        self.current_score = 0 
        self.best_score = 0
        self.num_episodes = 0

        self.reward = 0
        self.died = False

        self.plot = plot
        if self.plot: 
            self.score = []
            self.mean_score = []

    # one simulation step
    def move(self, action):
        
        # decoding the action
        if action == 0:
            self.force = constants.F
        else: 
            self.force = -constants.F

        temp = (self.force + constants.m * constants.l * self.angle**2 * math.sin(self.angle)) / (constants.M + constants.m)
        # pole angular acceleration
        angular_acc = (constants.g * math.sin(self.angle) - math.cos(self.angle) * temp) / (constants.l * (4/3 - constants.m * math.cos(self.angle)**2) / (constants.M + constants.m))
        # cart acceleration
        cart_acc = temp - constants.m * constants.l * angular_acc * math.cos(self.angle) / (constants.M + constants.m)

        # update the values
        self.velocity += cart_acc * constants.dt
        self.angular_velocity += angular_acc * constants.dt

        self.pos[0] += self.velocity * constants.dt
        self.angle += self.angular_velocity * constants.dt

        # score increases for staying alive
        self.current_score += 1

        # more vertical is the pole more reward
        self.reward = 1

        # agent did not die
        self.died = False
        
        # check if exceeded the limit
        self.reset()

        return self.reward, self.died

    # end the episode if exceed the limit
    def reset(self):
        if self.pos[0] < 0 or self.pos[0] > pos_lim or abs(self.angle) > angle_lim:
            self.pos[0] = settings.WIDTH/(2 * scaling)
            self.velocity = 0 
            self.angle = random.uniform(-init_angle, init_angle) 
            self.angular_velocity = 0 

            self.num_episodes += 1
            if self.current_score > self.best_score:
                self.best_score = self.current_score

            # calculate mean score and plot
            if self.plot:
                self.score.append(self.current_score)
                self.mean_score.append(sum(self.score)/len(self.score))
                plot_score(self.score, self.mean_score)

            self.current_score = 0

            # reward for exceeding the threshhold
            self.died = True
            self.reward = -1
    
    # get the current state
    def get_state(self):
        return [self.pos[0], self.velocity, self.angle, self.angular_velocity]
    
    # render the cartpole and score
    def draw(self, screen):
        # road
        pygame.draw.line(screen, settings.WHITE, (0, self.pos[1] + settings.CART_H/2), (settings.WIDTH, self.pos[1] + settings.CART_H/2), 2)

        # scaling back into pixels for rendering
        scaled_pos = self.pos[0] * scaling

        # cart
        pygame.draw.rect(screen, settings.RED, (scaled_pos, self.pos[1], settings.CART_W, settings.CART_H))

        # pole
        middle_x = scaled_pos + settings.CART_W / 2
        middle_y = self.pos[1] + settings.CART_H / 2

        pole_length = constants.l * scaling

        x = middle_x + pole_length * math.sin(self.angle)
        y = middle_y - pole_length * math.cos(self.angle)

        pygame.draw.line(screen, settings.GREEN, (middle_x, middle_y), (x, y), 3)
