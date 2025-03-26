from train import train_ddqn
from models.DDQN import DDQN_Agent
from logic.cartPole import CartPole

def main():

    agent = DDQN_Agent(input=4, plot=True)
    cartpole = CartPole(plot = True)

    train_ddqn(agent, cartpole)

if __name__ == "__main__":
    main()