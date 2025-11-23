# External libraries
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from collections import OrderedDict

# Local modules
from environment import PongEnviroment
from utils import parse_args, set_seed, plot_train_rewards, plot_evaluation_rewards

class Policy(nn.Module):
    def __init__(self, state_space: int, action_space: int, hidden_dim: int):
        super().__init__()
        # TODO: Define policy network architecture
        # Read the pytorch documentation on nn.Sequential to learn how to use it        
        self.network = nn.Sequential(OrderedDict([
            # TODO: Fill with the necessary layers
            ('policy_layer_1', nn.Linear(state_space, hidden_dim) ),
            ('policy_activation_1', nn.ReLU() ),
            ('policy_layer_2', nn.Linear(hidden_dim, hidden_dim) ),
            ('policy_activation_2', nn.ReLU() ),
            ('policy_layer_3', nn.Linear(hidden_dim, action_space) ),
        ]))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass
        return self.network(state)

class Value(nn.Module):
    def __init__(self, state_space: int, hidden_dim: int):
        super().__init__()
        # TODO: Define policy network architecture
        # Read the pytorch documentation on nn.Sequential to learn how to use it        
        self.network = nn.Sequential(OrderedDict([
            # TODO: Fill with the necessary layers
            ('value_layer_1', nn.Linear(state_space, hidden_dim) ),
            ('value_activation_1', nn.ReLU() ),
            ('value_layer_2', nn.Linear(hidden_dim, hidden_dim) ),
            ('value_activation_2', nn.ReLU() ),
            ('value_layer_3', nn.Linear(hidden_dim, 1) ),
        ]))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass
        return self.network(state)


class Agent(nn.Module):
    def __init__(
        self,
        state_space: int,
        action_space: int,
        gamma: float,
        lr: float,
        max_training_steps: int,
    ):
        """
        Initialize the agent.

        Parameters:
            state_space      (tuple) : Size of the state space
            action_space       (int) : Size of the action space
            gamma            (float) : Discount factor
            lr               (float) : Learning rate
            max_training_steps (int) : Max steps in training episodes
        """
        super(Agent, self).__init__()

        # Initialize parameters
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.lr = lr
        self.max_training_steps = max_training_steps

        # TODO: Initialize policy network (actor) 
        # Refer to the handout for the value of hidden_dim
        hidden_dim = 256 # requered by handout
        self.policy = Policy(state_space, action_space, hidden_dim)

        # TODO: Initialize value network (critic)
        # Refer to the handout for the value of hidden_dim
        self.value = Value(state_space, hidden_dim)

        # TODO: Initialize optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)

        # TODO: Precalculate cummulative discount factors 
        # [1, γ, γ^2, γ^3, ..., γ^max_steps]
        gammas = [gamma ** i for i in range(max_training_steps + 10)]
        self.cum_gamma = torch.tensor(gammas, dtype=torch.float32)

    def get_action(self, state: torch.Tensor) -> int:
        """
        Helper function to get the action the agent will take given the state
        of the environment.
        
        IMPORTANT: Should only be used when deploying the agent.

        Parameters:
            state (torch.Tensor) : State encoded as tensor.

        Returns:
            Returns an integer denoting the action the selected action.
        """
        # TODO: Sample action from the policy
        # HINT: Remember how to stop gradient propagation
        if state.dim() == 1:
            state = state.unsqueeze(0)   #(1, S)

        with torch.no_grad():
            logits = self.policy(state)   #(1, action_Space)
            probs = torch.softmax(logits, dim= -1)  #
            action = torch.multinomial(probs, 1)   #(1,1)

        return int(action.item())
        
    def n_step_returns(self, n: int, rewards: torch.Tensor, next_state_values: torch.Tensor, terminated: torch.Tensor) -> torch.Tensor:
        """
        Calculates the N-Step-Returns for every timestep

        n_step_returns_t = r_t + r_t+1 * gamma + r_t+2 * gamma ^ 2 + ... + r_t+n-1 * gamma ^ n-1 + V(s_t+n) * gamma ^ n
        
        Parameters:
            rewards (torch.Tensor): Rewards at timesteps t, t+1, ...
            next_state_values (torch.Tensor): Estimated value for states at timesteps t+1, t+2, ...
            terminated (torch.Tensor): True for the terminal states in next_states_values
        """
        # TODO: Calculate the N-Step-Returns
        # Hint: Use torch.Tensor.cumsum, torch.nn.functional.pad and self.cum_gamma for efficient implementation
        # IMPORTANT: Read the pytorch documentation for these functions carefully, especially for torch.nn.functional.pad
        
        # rewards: (N, T, 1)
        # next_values: (N, T, 1)
        # terminated: (N, T, 1)
        N, T, _ = rewards.shape
        device = rewards.device

        # pad rewards: (N, T+n-1, 1)
        rewards_pad = torch.nn.functional.pad(rewards, pad=(0, 0, 0, n - 1))

        # sliding windows: (N, T, 1, n)
        windows = rewards_pad.unfold(dimension=1, size=n, step=1)

        # gamma^k reshape → (1,1,1,n)
        gammas = self.cum_gamma[:n].to(device).view(1, 1, 1, n)

        # discounted sum over last dim
        discounted_sum = (windows * gammas).sum(dim=3)  # (N, T, 1)

        # bootstrap index = t + (n - 1)
        bootstrap_idx = torch.arange(T, device=device) + (n-1)
        bootstrap_idx = bootstrap_idx.clamp(max=T - 1)

        # V(s_{t+n})
        bootstrap_vals = next_state_values[:, bootstrap_idx, :]  # (N, T, 1)

        # terminal mask
        term_mask = terminated[:, bootstrap_idx, :]

        bootstrap_vals = bootstrap_vals * (~term_mask)

        # final return
        G = discounted_sum + (self.gamma ** n) * bootstrap_vals

        return G


    def value_loss(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the loss used to optimize the value network (critic)

        Parameters:
            states        (torch.Tensor): States at timesteps t, t+1, ...
            rewards       (torch.Tensor): Rewards at timesteps t, t+1, ...
            next_states   (torch.Tensor): States at timesteps t+1, t+2, ...
            terminated    (torch.Tensor): True for the terminal states in next_states
        """
        """
        states:      (N, T, S)
        rewards:     (N, T, 1)
        next_states: (N, T, S)
        terminated:  (N, T, 1)
        """
        # TODO: Calculate the loss for the value network (critic)
        # Hint: Remember how to stop gradient propagation for the target calculation
        # Refer to the handout for the value of n


        N, T, S = states.shape

        states_flat = states.view(N * T, S)
        next_states_flat = next_states.view(N * T, S)

        # V(s_t): need backprop
        values = self.value(states_flat).view(N, T, 1)  # (N, T, 1)

        n = 10
        with torch.no_grad():
            # V(s_{t+1})
            next_values = self.value(next_states_flat).view(N, T, 1)  # (N, T, 1)
            #  n_step_returns get G_t^(n)
            returns_n = self.n_step_returns(n, rewards, next_values, terminated)  # (N, T, 1)

        # MSE( V(s_t), G_t^(n) )
        loss = nn.functional.mse_loss(values, returns_n)
        return loss


    def update_value(self,) -> None:
        """
        Updates the parameters of the value network (critic) by performing
        a step of gradient descent and sets all gradients to zero.
        """
        # TODO: Use the value optimizer to update the parameters
        self.value_optimizer.step()
        # TODO: Use the value optimizer to set all the gradients to zero
        self.value_optimizer.zero_grad()



    def policy_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the loss used to optimize the policy network (actor)

        Parameters:
            states        (torch.Tensor): States at timesteps t, t+1, ...
            actions       (torch.Tensor): States at timesteps t, t+1, ...
            rewards       (torch.Tensor): Rewards at timesteps t, t+1, ...
            next_states   (torch.Tensor): States at timesteps t+1, t+2, ...
            terminated    (torch.Tensor): True for the terminal states in next_states
        """
        """
        states:      (N, T, S)
        actions:     (N, T, 1)
        rewards:     (N, T, 1)
        next_states: (N, T, S)
        terminated:  (N, T, 1)
        """
        # TODO: Calculate the loss for the policy network (actor)
        # Hint: Remember how to stop gradient propagation for the target calculation
        # Refer to the handout for the value of n
        # N, T, S = states.shape
        # B = N * T
        N, T, S = states.shape
        B = N * T

        states_flat = states.view(B, S)
        next_states_flat = next_states.view(B, S)

        n = 10
        with torch.no_grad():
            # V(s_t)
            values = self.value(states_flat).view(N, T, 1)          # (N, T, 1)
            # V(s_{t+1})
            next_values = self.value(next_states_flat).view(N, T, 1) # (N, T, 1)
            # G_t^(n)
            returns_n = self.n_step_returns(n, rewards, next_values, terminated)
            # A_t = G_t^(n) - V(s_t)
            advantages = returns_n - values                          # (N, T, 1)

        # log π_θ(a_t | s_t)
        logits = self.policy(states_flat)                             # (B, A)
        log_probs = nn.functional.log_softmax(logits, dim=-1)        # (B, A)

        actions_flat = actions.view(B).long()                        # (B,)
        advantages_flat = advantages.view(B)                         # (B,)

        selected_log_probs = log_probs[
            torch.arange(B, device=log_probs.device),
            actions_flat,
        ]  # (B,)

        # Policy loss = - E[ log π(a|s) * A ]
        loss = -(selected_log_probs * advantages_flat).mean()
        return loss
        
    def update_policy(self) -> None:
        """
        Updates the parameters of the policy network (actor) by performing
        a step of gradient descent and sets all gradients to zero.
        """
        # TODO: Use the policy optimizer to update the parameters
        self.policy_optimizer.step()
        # TODO: Use the policy optimizer to set all the gradients to zero
        self.policy_optimizer.zero_grad()


    def save(self, filename: str) -> None:
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "value_state_dict": self.value.state_dict(),
        }
        torch.save(checkpoint, filename)

    def load(self, filename: str) -> None:
        checkpoint = torch.load(filename, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.value.load_state_dict(checkpoint["value_state_dict"])

def deploy_agent(
    agent: Agent, env: PongEnviroment,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run the agent in the environment for one episode

    Parameters:
        agent    (Agent) : Reinforcement Learning Agent
        env (Environment): Environment

    Returns:
        state       (torch.FloatTensor) : States at each timestep with shape (N, T, S)
        action      (torch.IntTensor)   : Actions at each timestep with shape (N, T, 1)
        reward      (torch.FloatTensor) : Rewards after each action with shape (N, T, 1)
        new_state   (torch.FloatTensor) : New states after each action with shape (N, T, S)
        terminated  (torch.BoolTensor)  : Whether the new state is a terminal state (N, T, 1)
    """
    # TODO: Reset the environment
    # Hint: Read the environment API in environment.py
    # IMPORTANT: Do not pass a seed to the environment everytime you reset it
    # you should only call this once in main()

    state = env.reset()  # return numpy array shape (S,)

    # Start trajectory
    done = False

    states = []
    actions = []
    rewards = []
    new_states = []
    terminals = []


    while not done:
        # TODO: Get action from agent
        # HINT: Check the agent's methods
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        action = agent.get_action(state_tensor)

        # TODO: Take step in environment
        next_state, reward, terminated, truncated = env.step(action)

        # TODO: Store trajectory step
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        new_states.append(next_state)

        term_flag = bool(terminated)
        trunc_flag = bool(truncated)
        terminals.append(term_flag or trunc_flag)


        # TODO: Stop the trajectory if the episode got truncated or terminated
        # Hint: Read the environment.py documentation about truncated
        done = terminated or truncated
        state = next_state

    
    # IMPORTANT: Check the shapes and data types for the return values in the
    # docstring
    states_t = torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in states]).unsqueeze(0)
    actions_t = torch.tensor(actions, dtype=torch.int64).view(1, -1, 1)    # (1, T, 1)
    rewards_t = torch.tensor(rewards, dtype=torch.float32).view(1, -1, 1)  # (1, T, 1)
    new_states_t = torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in new_states]).unsqueeze(0)
    terminated_t = torch.tensor(terminals, dtype=torch.bool).view(1, -1, 1)    # (1, T, 1)

    return states_t, actions_t, rewards_t, new_states_t, terminated_t


def main():
    args = parse_args()

    if args.eval_only == False:
        set_seed(10301) # DON'T DELETE THIS

        # TODO: Initialize environment
        env = PongEnviroment(
            max_steps=args.max_steps,
            record=False
        )

        # Set random seed (DON'T DELETE THIS)
        env.reset(seed=10301)

        state_size = sum(env.observation_space.shape)
        action_size = int(env.action_space.n)

        # TODO: Initialize agent
        agent = Agent(
            state_space=state_size,
            action_space=action_size,
            gamma=args.gamma,
            lr=args.lr,
            max_training_steps=args.max_steps,
        )

        # Train the agent
        train_rewards_list = []
        eval_rewards_list = []
        for episode in tqdm(
            range(args.train_episodes), "Training episodes", leave=False
        ):
            # ============ Training ===========
            # TODO: Deploy the current policy to get a new trajectory
            states, actions, rewards, new_states, terminated = deploy_agent(agent, env)

            # Store training metrics
            train_mean_return = rewards.sum().item()
            train_rewards_list.append(train_mean_return)

            # TODO: Calculate the loss for the policy and value
            # Hint: Check the agent's methods
            value_loss = agent.value_loss(states, rewards, new_states, terminated)
            policy_loss = agent.policy_loss(states, actions, rewards, new_states, terminated)


            # Scale losses and get the gradients
            value_loss = value_loss / args.batch_size
            policy_loss = policy_loss / args.batch_size
            value_loss.backward()
            policy_loss.backward()

            if (episode + 1) % args.batch_size == 0:
                # TODO: Update the policy and value functions
                agent.update_value()
                agent.update_policy()

            # ============ Evaluation ===========
            if (episode + 1) % args.eval_every == 0:
                eval_mean_undiscounted_return = 0
                for eval_episode in range(args.eval_episodes):
                    # TODO: Deplay the agent
                    states, actions, rewards, new_states, terminated = deploy_agent(agent, env)

                    # Store evaluation metrics
                    T = rewards.shape[2]
                    eval_undiscounted_return = rewards.sum().item()
                    eval_mean_undiscounted_return += eval_undiscounted_return / args.eval_episodes
                eval_rewards_list.append(eval_mean_undiscounted_return)

            # Store network checkpoints
            if episode + 1 % args.store_every == 0:
                print("Storing checkpoint")
                agent.save("checkpoint.pth")
        
        # Store final network
        agent.save("checkpoint.pth")

        # Plot returns and moving average vs episodes
        train_rewards_array = np.array(train_rewards_list)
        eval_rewards_array = np.array(eval_rewards_list)
        plot_train_rewards(train_rewards_array)
        plot_evaluation_rewards(eval_rewards_array, args.eval_every)

    else:
        # TODO: Initialize environment with record = True
        env = PongEnviroment(
            max_steps=args.max_steps,
            record=True,       # record video
        )

        state_size = sum(env.observation_space.shape)
        action_size = int(env.action_space.n)
        
        # TODO: Initialize agent
        agent = Agent(
            state_space=state_size,
            action_space=action_size,
            gamma=args.gamma,
            lr=args.lr,
            max_training_steps=args.max_steps,
        )

        # Load networks
        agent.load("checkpoint.pth")

        # Evaluate the agent
        N = 20
        trajectory_indices = torch.arange(0, N)
        rewards_list = []
        for i in range(N):
            # TODO: Deploy the agent to get a trajectory
            states, actions, rewards, new_states, terminated = deploy_agent(agent, env)
            rewards_list.append(rewards)
        env.close()

        # Get the total reward for each trajectory
        rewards = torch.tensor([rewards.sum() for rewards in rewards_list])
        # Get the length of each trajectory
        steps = torch.tensor([rewards.shape[1] for rewards in rewards_list])

        # Identify the longest winning trajectory
        sorted_steps, sort_indices = steps.sort()
        rewards = rewards[sort_indices]
        trajectory_indices = trajectory_indices[sort_indices]
        longest_winning_trajectory = trajectory_indices[rewards > 0][-1].item()
        print(f"Longest winning trajectory: {longest_winning_trajectory}")

if __name__ == "__main__":
    main()
