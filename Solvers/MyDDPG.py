import random
from copy import deepcopy
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import Adam
from torch.distributions.normal import Normal

from Solvers.Abstract_Solver import AbstractSolver
# from lib import plotting


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, dtype=torch.float32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, dtype=torch.float32)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully Connected Layers for CNN output
        self.fc_cnn1 = nn.Linear(64 * 32 * 32, 512, dtype=torch.float32)
        self.fc_cnn2 = nn.Linear(512, hidden_sizes[1], dtype=torch.float32)

        sizes = [obs_dim + act_dim + hidden_sizes[1] + 3] + hidden_sizes + [1]
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, obs, act):
        x_image, x_position, x_deltas  = obs
        # Process RGB image through CNN layers
        x_image = self.pool(F.relu(self.conv1(x_image)))
        x_image = self.pool(F.relu(self.conv2(x_image)))
        x_image = x_image.view(x_image.size(0), -1)  # Flatten for FC layer
        x_image = F.relu(self.fc_cnn1(x_image))       # Hidden layer 1 for CNN output
        x_image = F.relu(self.fc_cnn2(x_image))       # Hidden layer 2 for CNN output

        x = torch.cat([x_image, x_position, x_deltas, act], dim=-1)
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x).squeeze(dim=-1)


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, act_lim, hidden_sizes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, dtype=torch.float32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, dtype=torch.float32)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully Connected Layers for CNN output
        self.fc_cnn1 = nn.Linear(64 * 32 * 32, 512, dtype=torch.float32)
        self.fc_cnn2 = nn.Linear(512, hidden_sizes[1], dtype=torch.float32)

        sizes = [obs_dim + hidden_sizes[1] + 3] + hidden_sizes + [act_dim]
        self.act_lim = act_lim
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, obs):
        x_image, x_position, x_deltas  = obs
        # Process RGB image through CNN layers
        x_image = self.pool(F.relu(self.conv1(x_image)))
        x_image = self.pool(F.relu(self.conv2(x_image)))
        x_image = x_image.view(x_image.size(0), -1)  # Flatten for FC layer
        x_image = F.relu(self.fc_cnn1(x_image))       # Hidden layer 1 for CNN output
        x_image = F.relu(self.fc_cnn2(x_image))       # Hidden layer 2 for CNN output

        x = torch.cat([x_image, x_position, x_deltas], dim=-1)
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.action_low + 0.5 * (self.action_high - self.action_low) * (torch.tanh(self.layers[-1](x)) + 1)


class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, bounds, hidden_sizes):
        super().__init__()
        self.q = QNetwork(obs_dim, act_dim, hidden_sizes)
        self.pi = PolicyNetwork(obs_dim, act_dim, bounds, hidden_sizes)


class MyDDPG(AbstractSolver):
    def __init__(self, env, eval_env, options):
        super().__init__(env, eval_env, options)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Create actor-critic network
        self.actor_critic = ActorCriticNetwork(
            env.observation_space_shape,
            env.action_space.shape[0],
            env.bounds,
            self.options.layers,
        ).to(self.device)
        # Create target actor-critic network
        self.target_actor_critic = deepcopy(self.actor_critic)

        self.policy = self.create_greedy_policy()

        self.optimizer_q = Adam(self.actor_critic.q.parameters(), lr=self.options.alpha)
        self.optimizer_pi = Adam(
            self.actor_critic.pi.parameters(), lr=self.options.alpha
        )

        # Freeze target actor critic network parameters
        for param in self.target_actor_critic.parameters():
            param.requires_grad = False

        # Replay buffer
        self.replay_memory = deque(maxlen=options.replay_memory_size)

    @torch.no_grad()
    def update_target_networks(self, tau=0.995):
        """
        Copy params from actor_critic to target_actor_critic using Polyak averaging.
        """
        for param, param_targ in zip(
            self.actor_critic.parameters(), self.target_actor_critic.parameters()
        ):
            param_targ.data.mul_(tau)
            param_targ.data.add_((1 - tau) * param.data)

    def create_greedy_policy(self):
        """
        Creates a greedy policy.

        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """

        @torch.no_grad()
        def policy_fn(state):
            state = torch.as_tensor(state, dtype=torch.float32)
            return self.actor_critic.pi(state).numpy()

        return policy_fn
    
    def preprocess_state(self,state, camera_resolution=(128, 128)):
        """
        Converts the Domain env observation state into a format compatible with the actrorNetwork.
        
        Args:
            state (dict): A dictionary with "attitude", "rgba_cam", and "target_deltas" keys.
            camera_resolution (tuple): Resolution of the camera image for resizing (default is 64x64).
            
        Returns:
            x_image (torch.Tensor): Preprocessed image tensor for input into the CNN.
            x_position (torch.Tensor): Flattened position data tensor for input into the FC layers.
        """
        # print(state)
        # Process the image data (rgba_cam)
        rgba_cam = state["rgba_cam"]
        # Convert from uint8 [0, 255] to float [0, 1]
        rgba_cam = rgba_cam.astype(np.float32) / 255.0
        # Discard alpha channel, assuming RGBA format
        rgb_cam = rgba_cam[:3, :, :]
        # print(rgba_cam.shape)
        # Ensure the image has the correct resolution
        rgb_cam = torch.tensor(rgb_cam).view(1, 3, *camera_resolution)  # Batch size of 1

        # Process the attitude data
        attitude = state["attitude"]#[:-11]
        attitude_tensor = torch.tensor(attitude, dtype=torch.float32).flatten()
        # print("attitude", state["attitude"])

        # Process the target deltas
        target_deltas = [torch.tensor(delta, dtype=torch.float32) for delta in state["target_deltas"]]
        target_deltas_tensor = torch.cat(target_deltas, dim=0).unsqueeze(0)
        # print("target_deltas", state["target_deltas"])
        # print(state["target_deltas"].shape)

        self.actionBuf.add(attitude_tensor, target_deltas_tensor.squeeze(0))
    
        # Get flattened replay memory tensor
        x_memory = self.actionBuf.get_memory_tensor().unsqueeze(0)
    

        # Concatenate attitude and target deltas into the position tensor
        x_position = torch.cat([attitude_tensor], dim=0).unsqueeze(0)  # Batch size of 1

        return rgb_cam.to(self.device), x_position.to(self.device), target_deltas_tensor.to(self.device), x_memory.to(self.device)


    @torch.no_grad()
    def select_action(self, state):
        """
        Selects an action given state.

         Returns:
            The selected action (as an int)
        """
        state = torch.as_tensor(state, dtype=torch.float32)
        mu = self.actor_critic.pi(state)
        m = Normal(
            torch.zeros(self.env.action_space.shape[0]),
            torch.ones(self.env.action_space.shape[0]),
        )
        noise_scale = 0.1
        action_limit = self.env.action_space.high[0]
        action = mu + noise_scale * m.sample()
        return torch.clip(
            action,
            -action_limit,
            action_limit,
        ).numpy()

    @torch.no_grad()
    def compute_target_values(self, next_states, rewards, dones):
        """
        Computes the target q values.

        Use:
            self.target_actor_critic.pi(states): Returns the greedy action at states.
            self.target_actor_critic.q(states, actions): Returns the Q-values 
                for (states, actions).

        Returns:
            The target q value (as a tensor).
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        policy_actions = self.target_actor_critic.pi(next_states)
        q_values = self.target_actor_critic.q(next_states, policy_actions)
        return rewards + ~dones.bool() * q_values * self.options.gamma


    def replay(self):
        """
        Samples transitions from the replay memory and updates actor_critic network.
        """
        if len(self.replay_memory) > self.options.batch_size:
            minibatch = random.sample(self.replay_memory, self.options.batch_size)
            minibatch = [
                np.array(
                    [
                        transition[idx]
                        for transition, idx in zip(minibatch, [i] * len(minibatch))
                    ]
                )
                for i in range(5)
            ]
            states, actions, rewards, next_states, dones = minibatch
            # Convert numpy arrays to torch tensors
            states = torch.as_tensor(states, dtype=torch.float32)
            actions = torch.as_tensor(actions, dtype=torch.float32)
            rewards = torch.as_tensor(rewards, dtype=torch.float32)
            next_states = torch.as_tensor(next_states, dtype=torch.float32)
            dones = torch.as_tensor(dones, dtype=torch.float32)

            # Current Q-values
            current_q = self.actor_critic.q(states, actions)
            # Target Q-values
            target_q = self.compute_target_values(next_states, rewards, dones)

            # Optimize critic network
            loss_q = self.q_loss(current_q, target_q).mean()
            self.optimizer_q.zero_grad()
            loss_q.backward()
            self.optimizer_q.step()

            # Optimize actor network
            loss_pi = self.pi_loss(states).mean()
            self.optimizer_pi.zero_grad()
            loss_pi.backward()
            self.optimizer_pi.step()

    def memorize(self, state, action, reward, next_state, done):
        """
        Adds transitions to the replay buffer.
        """
        self.replay_memory.append((state, action, reward, next_state, done))

    def train_episode(self):
        """
        Runs a single episode of the DDPG algorithm.

        Use:
            self.select_action(state): Sample an action from the policy.
            self.step(action): Performs an action in the env.
            self.memorize(state, action, reward, next_state, done): store the transition in
                the replay buffer.
            self.replay(): Sample transitions and update actor_critic.
            self.update_target_networks(): Update target_actor_critic using Polyak averaging.
        """

        state, _ = self.env.reset()
        for _ in range(self.options.steps):
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            action = self.select_action(state)
            next_state, reward, done, _ = self.step(action)
            self.memorize(state, action, reward, next_state, done)
            self.replay()
            if len(self.replay_memory) > self.options.batch_size:
                self.update_target_networks()
            if done:
                break
            state = next_state

    def q_loss(self, current_q, target_q):
        """
        The q loss function.

        args:
            current_q: Current Q-values.
            target_q: Target Q-values.

        Returns:
            The unreduced loss (as a tensor).
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        # print("CURRENT:",current_q)
        # print("TARGET:",target_q)
        # print("SUB:",current_q - target_q)
        q_loss = (current_q - target_q) ** 2
        # print(q_loss)
        return q_loss

    def pi_loss(self, states):
        """
        The policy gradient loss function.
        Note that you are required to define the Loss^PG
        which should be the integral of the policy gradient
        The "returns" is the one-hot encoded (return - baseline) value for each action a_t
        ('0' for unchosen actions).

        args:
            states:

        Use:
            self.actor_critic.pi(states): Returns the greedy action at states.
            self.actor_critic.q(states, actions): Returns the Q-values for (states, actions).

        Returns:
            The unreduced loss (as a tensor).
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        return -self.actor_critic.q(states, self.actor_critic.pi(states))

    def __str__(self):
        return "DDPG"