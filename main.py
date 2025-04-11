import argparse
from modes.train import train_agent
from modes.test import test_agent

def main():

    # terminal input
    parser = argparse.ArgumentParser(description="CartPoleAI - Train or Evaluate Agent")
    # mode
    parser.add_argument("--mode", choices=["train", "test"], required=True,
                        help="Select mode: 'train' to train the agent or 'test' to evaluate a trained model.")
    # algorithm
    parser.add_argument("--algo", choices=["ddqn", "ppo"], required=True,
                        help="Select algorithm: 'ddqn' or 'ppo'")
    # render
    parser.add_argument("--render", action="store_true", 
                        help="Enable rendering (used during training)")
    # path for testing
    parser.add_argument("--model_path", type=str, 
                        help="Path to the FOLDER of the trained model (required for test mode)")

    args = parser.parse_args()

    # training mode logic
    if args.mode == "train":
        if not args.algo:
            print("Error: Please specify an algorithm (ppo or ddqn) using --algo.")
            return
        print(f"Training {args.algo.upper()} agent...")
        train_agent(args.algo, render=args.render)

    # testing mode logic
    elif args.mode == "test":
        if not args.model_path:
            print("Error: --model_path is missing.")
            return
        print(f"Testing {args.algo.upper()} agent...")
        test_agent(args.algo, model_path=args.model_path)

if __name__ == "__main__":
    main()