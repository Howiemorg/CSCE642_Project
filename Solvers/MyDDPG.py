import random, os
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

def cuda_tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy()

class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, dtype=torch.float32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, dtype=torch.float32)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully Connected Layers for CNN output
        self.fc_cnn1 = nn.Linear(64 * 32 * 32, 512, dtype=torch.float32)
        self.fc_cnn2 = nn.Linear(512, 256, dtype=torch.float32)

        sizes = [obs_dim + act_dim + 256 + 3] + hidden_sizes + [1]
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, obs, act):
        x_image, x_position, x_deltas  = obs
        x_image = x_image.to(self.device)
        x_position = x_position.to(self.device)
        x_deltas = x_deltas.to(self.device)
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
    def __init__(self, obs_dim, act_dim, bounds, hidden_sizes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, dtype=torch.float32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, dtype=torch.float32)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully Connected Layers for CNN output
        self.fc_cnn1 = nn.Linear(64 * 32 * 32, 512, dtype=torch.float32)
        self.fc_cnn2 = nn.Linear(512, 256, dtype=torch.float32)

        sizes = [obs_dim + 256 + 3] + hidden_sizes + [act_dim]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.action_low = torch.tensor(bounds[0], dtype=torch.float32).to(self.device)
        self.action_high = torch.tensor(bounds[1], dtype=torch.float32).to(self.device)

        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, obs):
        x_image, x_position, x_deltas  = obs
        x_image = x_image.to(self.device)
        x_position = x_position.to(self.device)
        x_deltas = x_deltas.to(self.device)
        # Process RGB image through CNN layers
        x_image = self.pool(F.relu(self.conv1(x_image)))
        x_image = self.pool(F.relu(self.conv2(x_image)))
        x_image = x_image.view(x_image.size(0), -1)  # Flatten for FC layer
        x_image = F.relu(self.fc_cnn1(x_image))       # Hidden layer 1 for CNN output
        x_image = F.relu(self.fc_cnn2(x_image))       # Hidden layer 2 for CNN output

        x = torch.cat([x_image, x_position, x_deltas], dim=-1)
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.action_high * (F.tanh(self.layers[-1](x)))


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
        self.noise_scale = 0.4

        self.optimizer_q = Adam(self.actor_critic.q.parameters(), lr=self.options.critic_alpha)
        self.optimizer_pi = Adam(
            self.actor_critic.pi.parameters(), lr=self.options.actor_alpha
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
            state = self.preprocess_state(state)
            return self.actor_critic.pi(state).squeeze(0).cpu().numpy()

        return policy_fn
    
    def export_weights(self, prefix="642_"):
        weights_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../Weights/"
        )
        actor_dir =  os.path.join(
            weights_dir,
            prefix+"actor_network_weights.pth"
        )
        critic_dir =  os.path.join(
            weights_dir,
            prefix+"critic_network_weights.pth"
        )
        os.makedirs(weights_dir, exist_ok=True)
        torch.save(self.actor_critic.pi.state_dict(), actor_dir)
        torch.save(self.actor_critic.q.state_dict(), critic_dir)

    def load_weights(self, prefix="642_"):
        weights_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../Weights/"
        )
        actor_dir =  os.path.join(
            weights_dir,
            prefix+"actor_network_weights.pth"
        )
        critic_dir =  os.path.join(
            weights_dir,
            prefix+"critic_network_weights.pth"
        )

        if not os.path.exists(actor_dir) or not os.path.exists(critic_dir) :
            raise FileNotFoundError("Actor or critic weights not found")

        os.makedirs(weights_dir, exist_ok=True)
        self.actor_critic.pi.load_state_dict(torch.load(actor_dir))
        self.actor_critic.pi.eval()
        self.actor_critic.q.load_state_dict(torch.load(critic_dir))
        self.actor_critic.q.eval()
    
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

        # Concatenate attitude and target deltas into the position tensor
        x_position = torch.cat([attitude_tensor], dim=0).unsqueeze(0)  # Batch size of 1

        return rgb_cam.to(self.device), x_position.to(self.device), target_deltas_tensor.to(self.device)


    @torch.no_grad()
    def select_action(self, state):
        """
        Selects an action given state.

         Returns:
            The selected action (as an int)
        """
        mu = self.actor_critic.pi(state).squeeze(0)
        m = Normal(
            torch.zeros(self.env.action_space.shape[0], device=self.device),
            torch.ones(self.env.action_space.shape[0], device=self.device),
        )
        self.noise_scale = max(.1, self.noise_scale * .99)
        # print("Noise:", self.noise_scale)
        action_low_limit = torch.tensor(self.env.bounds[0], device=self.device)
        action_high_limit = torch.tensor(self.env.bounds[1], device=self.device)
        action = mu + self.noise_scale * m.sample()
        return torch.clip(
            action,
            action_low_limit,
            action_high_limit,
        ).cpu().numpy()

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
        return rewards + (1-dones) * q_values * self.options.gamma


    def replay(self):
        """
        Samples transitions from the replay memory and updates actor_critic network.
        """
        if len(self.replay_memory) > self.options.batch_size:
            # print("Replay")
            minibatch = random.sample(self.replay_memory, self.options.batch_size)
            minibatch = [
                np.array(
                    [
                        transition[idx]
                        for transition, idx in zip(minibatch, [i] * len(minibatch))
                    ]
                )
                for i in range(9)
            ]
            state_images, state_attitudes, state_deltas, actions, rewards, next_state_images, next_state_attitudes, next_state_deltas, dones = minibatch

            # print(state_images.shape)
            # print(state_attitudes.shape)
            # print(state_deltas.shape)
            # Convert numpy arrays to torch tensors
            state_images = torch.as_tensor(state_images, dtype=torch.float32, device=self.device)
            state_attitudes = torch.as_tensor(state_attitudes, dtype=torch.float32, device=self.device)
            state_deltas = torch.as_tensor(state_deltas, dtype=torch.float32, device=self.device)
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
            rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
            next_state_images = torch.as_tensor(next_state_images, dtype=torch.float32, device=self.device)
            next_state_attitudes = torch.as_tensor(next_state_attitudes, dtype=torch.float32, device=self.device)
            next_state_deltas = torch.as_tensor(next_state_deltas, dtype=torch.float32, device=self.device)
            dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

            # Current Q-values
            current_q = self.actor_critic.q((state_images, state_attitudes, state_deltas), actions)
            # Target Q-values
            target_q = self.compute_target_values((next_state_images, next_state_attitudes, next_state_deltas), rewards, dones)

            # Optimize critic network
            loss_q = self.q_loss(current_q, target_q)
            self.optimizer_q.zero_grad()
            loss_q.backward()
            self.optimizer_q.step()

            # Optimize actor network
            loss_pi = self.pi_loss((state_images, state_attitudes, state_deltas)).mean()
            self.optimizer_pi.zero_grad()
            loss_pi.backward()
            self.optimizer_pi.step()

            #print(f"Loss Q: {loss_q.item():.3f}, Loss Pi: {loss_pi.item():.3f}")

    def memorize(self, state, action, reward, next_state, done):
        """
        Adds transitions to the replay buffer.
        """
        state = [cuda_tensor_to_numpy(tensor).squeeze(0) for tensor in state]
        next_state = [cuda_tensor_to_numpy(tensor).squeeze(0) for tensor in next_state]
        self.replay_memory.append((state[0], state[1], state[2], action, reward, next_state[0], next_state[1], next_state[2], done))

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
        state = self.preprocess_state(state)
        for _ in range(self.options.steps):
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            action = self.select_action(state)
            next_state, reward, done, _ = self.step(action)
            next_state = self.preprocess_state(next_state)
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
        q_loss =  nn.MSELoss()(current_q, target_q)
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