import random
import math
import config.display as display
import config.constants as constants
import pygame
from utils.plot import plot_score
from collections import deque

class CartPole:
    def __init__(self, plot:bool):
        self.pos = [display.WIDTH/(2 * display.scaling), 2/3 * display.HEIGHT] # cart's initial postion, in the middle of line (x-axis scaled into meters)
        self.velocity = 0 # cart's initial velocity
        self.angle = random.uniform(-constants.init_angle, constants.init_angle) # pole's initial angle

        self.angular_velocity = 0 # pole's initial angular velocity
        self.force = 0 # initial force applied to the cart

        self.current_score = 0 
        self.best_score = 0
        self.num_episodes = 0

        # plot?
        self.plot = plot
        if plot: 
            self.score = []
            self.mean_score = []

        # track mean of last x episodes -> when to save model
        self.score_hist = deque(maxlen=100)
        self.old_mean = 0
        self.new_mean = 0

        self.reward = 0
        self.died = False

    # one simulation step
    def move(self, action):
        
        # decoding the action
        if action == 0:
            self.force = constants.F # appy force from right
        else: 
            self.force = -constants.F # apply force from left

        temp = (self.force + constants.m * constants.l * self.angle**2 * math.sin(self.angle)) / (constants.M + constants.m)
        # pole angular acceleration
        angular_acc = (constants.g * math.sin(self.angle) - math.cos(self.angle) * temp) / (constants.l * (4/3 - constants.m * math.cos(self.angle)**2) / (constants.M + constants.m))
        # cart acceleration
        cart_acc = temp - constants.m * constants.l * angular_acc * math.cos(self.angle) / (constants.M + constants.m)

        # update the values
        self.velocity += cart_acc * constants.dt
        self.angular_velocity += angular_acc * constants.dt

        # apply friction
        self.velocity *= (1 - constants.c)
        self.angular_velocity *= (1 - constants.p)

        self.pos[0] += self.velocity * constants.dt
        self.angle += self.angular_velocity * constants.dt

        # score increases for staying alive
        self.current_score += 1

        # more vertical is the pole -> more reward
        self.reward = 1 - abs(self.angle/constants.angle_lim)

        # agent did not die
        self.died = False

        # check for reset
        self.reset()

        return self.reward, self.died

    # end the episode if exceed the limit
    def reset(self):
        if self.pos[0] < 0 or self.pos[0] > constants.pos_lim or abs(self.angle) > constants.angle_lim:
            self.pos[0] = display.WIDTH/(2 * display.scaling)
            self.velocity = 0 
            self.angle = random.uniform(-constants.init_angle, constants.init_angle) 
            self.angular_velocity = 0 

            self.num_episodes += 1

            # best score
            if self.current_score > self.best_score:
                self.best_score = self.current_score

            # update score history
            self.old_mean = self.new_mean
            self.score_hist.append(self.current_score)
            self.new_mean = sum(self.score_hist)/len(self.score_hist)

            # add to the plotting data
            if self.plot:
                self.score.append(self.current_score)
                self.mean_score.append(sum(self.score)/len(self.score))

            self.current_score = 0

            # reward for exceeding the threshhold
            self.died = True
            self.reward = -5
    
    # get the current state
    def get_state(self):
        return [self.pos[0], self.velocity, self.angle, self.angular_velocity]
    
    # render the cartpole and score
    def draw(self, screen):
        # road
        pygame.draw.line(screen, display.WHITE, (0, self.pos[1] + display.CART_H/2), (display.WIDTH, self.pos[1] + display.CART_H/2), 2)

        # scaling back into pixels for rendering
        scaled_pos = self.pos[0] * display.scaling

        # cart
        pygame.draw.rect(screen, display.RED, (scaled_pos, self.pos[1], display.CART_W, display.CART_H))

        # pole
        middle_x = scaled_pos + display.CART_W / 2
        middle_y = self.pos[1] + display.CART_H / 2

        pole_length = constants.l * display.scaling

        x = middle_x + pole_length * math.sin(self.angle)
        y = middle_y - pole_length * math.cos(self.angle)

        pygame.draw.line(screen, display.GREEN, (middle_x, middle_y), (x, y), 3)

    # plot the current and mean score
    def plot_score(self):
        if self.plot and len(self.score) >= 1:
            plot_score(self.score, self.mean_score)