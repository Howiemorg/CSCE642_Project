import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import Adam

from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [act_dim]
        self.layers = nn.ModuleList()
        
        total_params_amt = 0
        # Layers
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            total_params_amt += sizes[i] * sizes[i + 1] + sizes[i + 1]

        self.total_params_amt = total_params_amt

    def forward(self, obs):
        x = torch.cat([obs], dim=-1)
        for i in range(len(self.layers) - 2):
            x = F.relu(self.layers[i](x))
        # Actor head
        probs = F.softmax(self.layers[-1](x), dim=-1)
        print("Probs from Actor:", probs)
        return torch.squeeze(probs, -1)


class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_sizes):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [1]
        self.layers = nn.ModuleList()
        
        total_params_amt = 0
        # Layers
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            total_params_amt += sizes[i] * sizes[i + 1] + sizes[i + 1]
            
        self.total_params_amt = total_params_amt

    def forward(self, obs):
        x = torch.cat([obs], dim=-1)
        for i in range(len(self.layers) - 2):
            x = F.relu(self.layers[i](x))
        # Critic head
        value = self.layers[-1](x)

        return torch.squeeze(value, -1)


class A2CEligibility(AbstractSolver):
    def __init__(self, env, eval_env, options):
        super().__init__(env, eval_env, options)
        # Create actor-critic network
        self.actor = ActorNetwork(
            env.observation_space.shape[0], env.action_space.n, self.options.layers
        )
        self.policy = self.create_greedy_policy()

        self.actor_optimizer = Adam(
            self.actor.parameters(), lr=self.options.actor_alpha)
        
        self.z_actor = torch.zeros(self.actor.total_params_amt)
        
        self.critic = CriticNetwork(
            env.observation_space.shape[0], self.options.layers
        )

        self.critic_optimizer = Adam(
            self.critic.parameters(), lr=self.options.critic_alpha)
        
        self.z_critic = torch.zeros(self.critic.total_params_amt)

    def create_greedy_policy(self):
        """
        Creates a greedy policy.


        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """

        def policy_fn(state):
            state = torch.as_tensor(state, dtype=torch.float32)
            return torch.argmax(self.actor(state)).detach().numpy()

        return policy_fn

    def select_action(self, state):
        """
        Selects an action given state.

        Returns:
            The selected action (as an int)
            The probability of the selected action (as a tensor)
            The critic's value estimate (as a tensor)
        """
        state = torch.as_tensor(state, dtype=torch.float32)
        probs = self.actor(state)
        value = self.critic(state)

        probs_np = probs.detach().numpy()
        action = np.random.choice(len(probs_np), p=probs_np)

        return action, probs[action], value
    
    def update_critic(self, advantage):
        for param, eligibility_trace in zip(self.critic.parameters(), self.z_critic):
            eligibility_trace = self.options.gamma * self.options.critic_trace_decay * eligibility_trace + param.grad
            param.grad = eligibility_trace * advantage
            
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()
        
    def update_actor(self, advantage, prob):
        loss = -torch.log(prob) * advantage
        
        self.actor_optimizer.zero_grad()
        # print("Before backward - grad of actor parameters:",
        #       [param.grad for param in self.actor.parameters()])
        loss.backward()
        # print("After backward - grad of actor parameters:",
        #       [param.grad for param in self.actor.parameters()])

        for param, eligibility_trace in zip(self.actor.parameters(), self.z_actor):
            eligibility_trace = self.options.gamma * \
                self.options.actor_trace_decay * eligibility_trace + param.grad
            param.grad = eligibility_trace * advantage

        self.actor_optimizer.step()
        

    def update_actor_critic(self, advantage, prob):
        """
        Performs actor critic update.

        args:
            advantage: Advantage of the chosen action (tensor).
            prob: Probability associated with the chosen action (tensor).
            value: Critic's state value estimate (tensor).
        """
        # Compute loss
        self.update_actor(advantage, prob)
        self.update_critic(advantage)

    def train_episode(self):
        """
        Run a single episode of the A2C algorithm.

        Use:
            self.select_action(state): Sample an action from the policy.
            self.step(action): Perform an action in the env.
            self.options.gamma: Gamma discount factor.
            self.actor_critic(state): Returns the action probabilities and
                the critic's estimate at a given state.
            torch.as_tensor(state, dtype=torch.float32): Converts a numpy array
                'state' to a tensor.
            self.update_actor_critic(advantage, prob, value): Update actor critic. 
        """

        state, _ = self.env.reset()
        self.z_actor = self.z_actor.zero_()
        self.z_critic = self.z_critic.zero_()
        for _ in range(self.options.steps):
            action, action_prob, estimate = self.select_action(state)
            next_state, reward, done, _ = self.step(action)
            advantage = reward - estimate
            next_state_tensor = torch.as_tensor(
                next_state, dtype=torch.float32)
            value = self.critic(next_state_tensor)
            advantage += (self.options.gamma * value)      
            
            self.update_actor_critic(advantage, action_prob)
            if (done):
                break
            state = next_state

    def actor_loss(self, advantage, prob):
        """
        The policy gradient loss function.
        Note that you are required to define the Loss^PG
        which should be the integral of the policy gradient.

        args:
            advantage: Advantage of the chosen action.
            prob: Probability associated with the chosen action.

        Use:
            torch.log: Element-wise logarithm.

        Returns:
            The unreduced loss (as a tensor).
        """
        return -advantage * torch.log(prob)

    def __str__(self):
        return "A2C"

    def plot(self, stats, smoothing_window=20, final=False):
        plotting.plot_episode_stats(stats, smoothing_window, final=final)
