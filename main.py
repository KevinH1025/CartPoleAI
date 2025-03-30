from models.DDQN import DDQN_Agent
from cartPole import CartPole
from train import train_ddqn
from test import test_agent

def main():

    #agent = DDQN_Agent(input=4, plot=True)
    #cartpole = CartPole(plot = True)
    
    #train_ddqn(agent, cartpole)

    test_agent("ddqn")

if __name__ == "__main__":
    main()