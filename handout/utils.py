import random
import numpy as np
import torch
from argparse import ArgumentParser
import matplotlib.pyplot as plt


def set_seed(seed):
    """
    DO NOT MODIFY THIS FUNCTION.
    Sets random seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    """
    DO NOT MODIFY THIS FUNCTION.
    Parses all command-line arguments and returns them.

    Returns:
        args : An argparse Namespace object containing the following attributes:
            max_steps (int): Maximum steps per episode (default: 300)
            gamma (float): Discount factor (default: 0.98)
            train_episodes (int): Total number of training episodes (default: 20000)
            lr (float): Learning rate (default: 0.0003)
            batch_size (int): Episodes for each policy and value network update (default: 4)
            store_every (int): Episodes between each checkpoint (default: 2000)
            eval_only (bool): Flag to record a video of agent playing (default: False)
            eval_every (int): Training episodes between evaluation episodes (default: 500)
            eval_episodes (int): Number of episodes per evaluation (default: 20)
    """
    parser = ArgumentParser()
    # Environment settings
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="Maximum steps per episode",
    )

    # Agent hyperparameters
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.98,
        help="Discount factor",
    )

    # Training
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=30000,
        help="Total number of training episodes",
    )
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Episodes for each policy and value network update",
    )
    parser.add_argument(
        "--store-every",
        type=int,
        default=2000,
        help="Episodes between each checkpoint",
    )

    # Evaluation
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Records a video of your agent playing Pong",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=500,
        help="Training episodes in between evaluation episodes",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="Episodes at each evaluation",
    )

    args = parser.parse_args()
    return args


def plot_train_rewards(rewards: np.ndarray):
    plt.plot(rewards, label="Reward per Episode")
    window_size = 50
    rolling_mean = moving_average(rewards, n=window_size)
    plt.plot(rolling_mean, label=f"Rolling Mean (window = {window_size})")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Training Reward by Episode")
    plt.legend()
    plt.savefig("training_rewards.jpeg")
    plt.close()
    return


def plot_evaluation_rewards(rewards, training_episodes_between_evaluations):
    episodes = np.arange(1, len(rewards) + 1) * training_episodes_between_evaluations
    plt.plot(episodes, rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Evaluation Rewards by Episode")
    plt.savefig("evaluation_rewards.jpeg")
    plt.close()
    return


# Helper function to calculate the moving average of the returns
def moving_average(arr, n=5):
    average = np.cumsum(arr)
    average[n:] = (average[n:] - average[:-n]) / n
    average[:n] = average[:n] / (1 + np.arange(n))
    return average
