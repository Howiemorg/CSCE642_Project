# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) inform Guni Sharon at 
# guni@tamu.edu regarding your usage (relevant statistics is reported to NSF).
# The development of this assignment was supported by NSF (IIS-2238979).
# Contributors:
# The core code base was developed by Guni Sharon (guni@tamu.edu).
# The PyTorch code was developed by Sheelabhadra Dey (sheelabhadra@tamu.edu).

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

class MDPBuffer:
    def __init__(self, replay_memory_size, state_dim):
        self.replay_memory_size = replay_memory_size
        self.buffer = []  # List to hold memory entries
        self.state_dim = state_dim  # Dimension of each [attitude, target_deltas] entry
    
    def add(self, attitude, target_deltas):
        # Concatenate attitude and target_deltas and add to memory
        memory_entry = torch.cat((attitude, target_deltas), dim=0)
        
        # Maintain buffer size
        if len(self.buffer) >= self.replay_memory_size:
            self.buffer.pop(0)
        
        self.buffer.append(memory_entry)
    
    def get_memory_tensor(self):
        # Pad memory if it has fewer than replay_memory_size items
        if len(self.buffer) < self.replay_memory_size:
            padding = [torch.zeros(self.state_dim)] * (self.replay_memory_size - len(self.buffer))
            # print(padding)
            memory_stack = torch.stack(self.buffer + padding)
        else:
            memory_stack = torch.stack(self.buffer)
        
        # Flatten the memory stack into a single tensor
        return memory_stack.view(-1)

class QNetwork(nn.Module):
    # def __init__(self, obs_dim, act_dim, hidden_sizes):
    #     super().__init__()
    #     sizes = [obs_dim + act_dim] + hidden_sizes + [1]
    #     self.layers = nn.ModuleList()
    #     for i in range(len(sizes) - 1):
    #         self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def __init__(self, obs_dim, act_dim, hidden_size, replay_size):
        super().__init__()
              
        # RGB Image Branch (CNN)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, dtype=torch.float32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, dtype=torch.float32)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully Connected Layers for CNN output
        self.fc_cnn1 = nn.Linear(64 * 32 * 32, 512, dtype=torch.float32)
        self.fc_cnn2 = nn.Linear(512, hidden_size[1], dtype=torch.float32)
        
        # Fully Connected Branch for Position Record (e.g., Current and Past Position)
        self.fc_position1 = nn.Linear(obs_dim, hidden_size[0], dtype=torch.float32)
        self.fc_position2 = nn.Linear(hidden_size[0], hidden_size[1], dtype=torch.float32)

        #Fully Connected Branch for Position Record (e.g., Current and Past Position)
        self.fc_delta1 = nn.Linear(3, hidden_size[0], dtype=torch.float32)
        self.fc_delta2 = nn.Linear(hidden_size[0], hidden_size[1], dtype=torch.float32)

        self.fc_memory1 = nn.Linear(replay_size, hidden_size[0])
        self.fc_memory2 = nn.Linear(hidden_size[0], hidden_size[1])

        self.fc_act1 = nn.Linear(act_dim, hidden_size[0])
        self.fc_act2= nn.Linear(hidden_size[0], hidden_size[1])
        
        # Final output layer to predict the value, combining CNN (256) and position (64) outputs
        self.fc_value = nn.Linear(hidden_size[1]*5, 1, dtype=torch.float32)

        # self.action_low = torch.tensor(bounds[0])
        # self.action_high = torch.tensor(bounds[1])

        self.apply(self.initialize_weights)
    
    def initialize_weights(self, layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        
        elif isinstance(layer, nn.Linear):
            if layer == self.fc_value:
                # Initialize the output layers with smaller variance
                nn.init.uniform_(layer.weight, -3e-3, 3e-3)
                nn.init.constant_(layer.bias, 0)
            else:
                # Use Xavier initialization for non-output FC layers
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    # def forward(self, obs, act):
    #     x = torch.cat([obs, act], dim=-1)
    #     for i in range(len(self.layers) - 1):
    #         x = F.relu(self.layers[i](x))
    #     return self.layers[-1](x).squeeze(dim=-1)

    def forward(self, x, act):
        x_image, x_position, x_deltas, x_memory  = x
        # Process RGB image through CNN layers
        x_image = self.pool(F.relu(self.conv1(x_image)))
        x_image = self.pool(F.relu(self.conv2(x_image)))
        x_image = x_image.view(x_image.size(0), -1)  # Flatten for FC layer
        x_image = F.relu(self.fc_cnn1(x_image))       # Hidden layer 1 for CNN output
        x_image = F.relu(self.fc_cnn2(x_image))       # Hidden layer 2 for CNN output

        # Process position information through FC layers
        x_position = F.relu(self.fc_position1(x_position))  # Hidden layer 1 for position
        x_position = F.relu(self.fc_position2(x_position))  # Hidden layer 2 for position

        x_deltas = F.relu(self.fc_delta1(x_deltas))  # Hidden layer 1 for position
        x_deltas = F.relu(self.fc_delta2(x_deltas))  # Hidden layer 2 for position

        x_memory = F.relu(self.fc_memory1(x_memory))
        x_memory = F.relu(self.fc_memory2(x_memory))

        
        if(len(act.shape) < 2):
            act = act.unsqueeze(0)
        act = F.relu(self.fc_act1(act))
        act = F.relu(self.fc_act2(act))

        # Concatenate image and position features
        x_combined = torch.cat([x_image, x_position, x_deltas, x_memory, act], dim=-1)
        
        # Output the value prediction
        value = self.fc_value(x_combined)
        return torch.squeeze(value, -1)



class PolicyNetwork(nn.Module):
    # def __init__(self, obs_dim, act_dim, act_lim, hidden_sizes):
    #     super().__init__()
    #     sizes = [obs_dim] + hidden_sizes + [act_dim]
    #     self.act_lim = act_lim
    #     self.layers = nn.ModuleList()
    #     for i in range(len(sizes) - 1):
    #         self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def __init__(self, obs_dim, act_dim, bounds, hidden_size, replay_size):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, dtype=torch.float32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, dtype=torch.float32)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully Connected Layers for CNN output
        self.fc_cnn1 = nn.Linear(64 * 32 * 32, 512, dtype=torch.float32)
        self.fc_cnn2 = nn.Linear(512, hidden_size[1], dtype=torch.float32)
        
        # Fully Connected Branch for Position Record (e.g., Current and Past Position)
        self.fc_position1 = nn.Linear(obs_dim, hidden_size[0], dtype=torch.float32)
        self.fc_position2 = nn.Linear(hidden_size[0], hidden_size[1], dtype=torch.float32)

        # Fully Connected Branch for Position Record (e.g., Current and Past Position)
        self.fc_delta1 = nn.Linear(3, hidden_size[0], dtype=torch.float32)
        self.fc_delta2 = nn.Linear(hidden_size[0], hidden_size[1], dtype=torch.float32)

        self.fc_memory1 = nn.Linear(replay_size, hidden_size[0], dtype=torch.float32)
        self.fc_memory2 = nn.Linear(hidden_size[0], hidden_size[1], dtype=torch.float32)

        # Final hidden layers of combined inputs
        # self.hidden1 = nn.Linear(hidden_size[1]*4, hidden_size[0], dtype=torch.float32)
        # self.hidden2 = nn.Linear(hidden_size[0], hidden_size[1], dtype=torch.float32)        

        self.final = nn.Linear(hidden_size[1]*4, act_dim, dtype=torch.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.action_low = torch.tensor(bounds[0], dtype=torch.float32).to(device)
        self.action_high = torch.tensor(bounds[1], dtype=torch.float32).to(device)

        self.apply(self.initialize_weights)
    
    def initialize_weights(self, layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        
        elif isinstance(layer, nn.Linear):
            if layer == self.final:
                # Initialize the output layers with smaller variance
                nn.init.uniform_(layer.weight, -3e-3, 3e-3)
                nn.init.constant_(layer.bias, 0)
            else:
                # Use Xavier initialization for non-output FC layers
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    # def forward(self, obs):
    #     x = torch.cat([obs], dim=-1)
    #     for i in range(len(self.layers) - 1):
    #         x = F.relu(self.layers[i](x))
    #     return self.act_lim * F.tanh(self.layers[-1](x))

    def forward(self, x):
        x_image, x_position, x_deltas, x_memory = x
        # print("Image shape:", x_image.shape)
        # print("Position shape:", x_position.shape)
    
        # Process RGB image
        x_image = self.pool(F.relu(self.conv1(x_image)))
        # print(x_image.shape)
        x_image = self.pool(F.relu(self.conv2(x_image)))
        # print(x_image.shape)
        x_image = x_image.view(x_image.size(0), -1)  # Flatten for FC layer
        # print(x_image.shape)
        x_image = F.relu(self.fc_cnn1(x_image))     # Hidden layer 1 for CNN output
        x_image = F.relu(self.fc_cnn2(x_image))     # Hidden layer 2 for CNN output
        # print("after cnn hidden layers")
        # Process position history
        x_position = F.relu(self.fc_position1(x_position))  # Hidden layer 1 for position
        x_position = F.relu(self.fc_position2(x_position))  # Hidden layer 2 for position

        x_deltas = F.relu(self.fc_delta1(x_deltas))  # Hidden layer 1 for position
        x_deltas = F.relu(self.fc_delta2(x_deltas))  # Hidden layer 2 for position

        x_memory = F.relu(self.fc_memory1(x_memory))
        x_memory = F.relu(self.fc_memory2(x_memory))

        x_combined = torch.cat([x_image, x_position, x_deltas, x_memory], dim=-1)
        
        # x_combined = F.relu(self.hidden1(x_combined))
        # x_combined = F.relu(self.hidden2(x_combined))

        actions = self.final(x_combined).squeeze(dim=-1)

        return actions


class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, bounds, hidden_size, replay_size):
        super().__init__()
        self.q = QNetwork(obs_dim, act_dim, hidden_size, replay_size)
        self.pi = PolicyNetwork(obs_dim, act_dim, bounds, hidden_size, replay_size)


class DDPG(AbstractSolver):
    def __init__(self, env, eval_env, options):
        super().__init__(env, eval_env, options)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.actionBuf = MDPBuffer(self.options.mdp_buff_size, env.observation_space_shape+3)


        # Create actor-critic network
        self.actor_critic = ActorCriticNetwork(
            env.observation_space_shape,
            env.action_space.shape[0],
            env.bounds, self.options.layers, self.actionBuf.state_dim* self.options.mdp_buff_size
        ).to(self.device)
        # Create target actor-critic network
        self.target_actor_critic = deepcopy(self.actor_critic).to(self.device)

        self.policy = self.create_greedy_policy()

        self.optimizer_q = Adam(self.actor_critic.q.parameters(), lr=self.options.alpha)
        self.optimizer_pi = Adam(
            self.actor_critic.pi.parameters(), lr=self.options.alpha
        )

      
        # Freeze target actor critic network parameters
        for param in self.target_actor_critic.parameters():
            param.requires_grad = False
            # nn.utils.clip_grad_norm_(param, max_norm=1.0)

        # Replay buffer
        # self.actionBuf = MDPBuffer(self.options.replay_memory_size, env.observation_space_shape+3)
        self.replay_memory = deque(maxlen=self.options.replay_memory_size)

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
        # state = torch.as_tensor(state, dtype=torch.float32)
        state = self.preprocess_state(state)
        mu = self.actor_critic.pi(state).squeeze(0)
        m = Normal(
            torch.zeros(self.env.action_space.shape[0]),
            torch.ones(self.env.action_space.shape[0]),
        )
        noise_scale = 0.1
        action_low_limit = torch.tensor(self.env.bounds[0]).to(self.device)
        action_high_limit = torch.tensor(self.env.bounds[1]).to(self.device)
        noise = torch.randn_like(mu) * 0.1
        # action = mu +noise
        action = (mu + noise).clamp(action_low_limit, action_high_limit)
        return action.cpu().numpy()
        return torch.clip(
            action,
            action_low_limit,
            action_high_limit,
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
        policy_actions = self.target_actor_critic.pi(next_states)
        q_values = self.target_actor_critic.q(next_states, policy_actions)
        return rewards + (1-dones) * q_values * self.options.gamma




    def update_actor_critic(self, state, action, reward, next_state, done):
        """
        Samples transitions from the replay memory and updates actor_critic network.
        """

        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(reward, dtype=torch.float32)
        done = torch.as_tensor(done, dtype=torch.float32)

        # Current Q-values
        current_q = self.actor_critic.q(state, action)
        # Target Q-values
        target_q = self.compute_target_values(next_state, reward, done)

        # Optimize critic network
        loss_q = self.q_loss(current_q, target_q)
        self.optimizer_q.zero_grad()
        loss_q.backward()
        self.optimizer_q.step()

        # Optimize actor network
        loss_pi = self.pi_loss(state).mean()
        self.optimizer_pi.zero_grad()
        loss_pi.backward()
        self.optimizer_pi.step()

    
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
        # state = self.preprocess_state(state)
        for n in range(self.options.steps):
            
            action = self.select_action(state)
            next_state, reward, done, _ = self.step(action)
            # next_state = self.preprocess_state(next_state)
            # self.memorize(state, action, reward, next_state, done)
            # self.replay()
            # if len(self.replay_memory) > self.options.batch_size:
            #     self.update_target_networks()
            # if done:
            #     break
            # self.update_actor_critic(state, action, reward, next_state, done)

            # self.update_target_networks()
            # state = next_state

            self.memorize(state, action, reward, next_state, done)
            if done:
                break

            if len(self.replay_memory) > self.options.batch_size and (n%5 ==0 and n!=0):
                self.replay()
                self.update_target_networks()


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

        # print("CURRENT:",current_q)
        # print("TARGET:",target_q)
        # print("SUB:",current_q - target_q)
        # q_loss = (current_q - target_q) ** 2
        # print(q_loss)
        return F.mse_loss(current_q, target_q)
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
        return -self.actor_critic.q(states, self.actor_critic.pi(states))

    def __str__(self):
        return "DDPG"
    
    # def replay(self):
    #     """
    #     Samples transitions from the replay memory and updates actor_critic network.
    #     """
    #     if len(self.replay_memory) < self.options.batch_size:
    #         return
    #     batch = random.sample(self.replay_memory, self.options.batch_size)
    #     states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
    #     self.update_actor_critic(states, actions, rewards, next_states, dones)
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
            # vfunc = np.vectorize(self.preprocess_state)
            states = [
                self.preprocess_state(s)
                for s in states
            ]
            # for components in zip(*states):
            #     print(torch.cat(components).shape)
            states = tuple([torch.cat(components) for components in zip(*states)] )
            actions = torch.as_tensor(actions, dtype=torch.float32).to(self.device)
            rewards = torch.as_tensor(rewards, dtype=torch.float32).to(self.device)
            next_states = [
                self.preprocess_state(s)
                for s in next_states
            ]
            next_states = tuple([torch.cat(components) for components in zip(*next_states)] )
            dones = torch.as_tensor(dones, dtype=torch.float32).to(self.device)

            # Current Q-values
            current_q = self.actor_critic.q(states, actions)
            # Target Q-values
            with torch.no_grad():
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
            print(f"Loss Q: {loss_q.item():.3f}, Loss Pi: {loss_pi.item():.3f}")


    def memorize(self, state, action, reward, next_state, done):
        """
        Adds transitions to the replay buffer.
        """
        self.replay_memory.append((state, action, reward, next_state, done))

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


    # def plot(self, stats, smoothing_window=20, final=False):
    #     plotting.plot_episode_stats(stats, smoothing_window, final=final)