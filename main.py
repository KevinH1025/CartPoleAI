from train import train_ddqn
from DDQN import DDQN_Agent
from cartPole import CartPole

def main():

    agent = DDQN_Agent(input=4)
    cartpole = CartPole()

    train_ddqn(agent, cartpole)

if __name__ == "__main__":
    main()