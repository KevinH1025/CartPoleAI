from modes.train import train_agent
from modes.test import test_agent

def main():
    
    train_agent("ddqn")

    #test_agent("ddqn")

if __name__ == "__main__":
    main()