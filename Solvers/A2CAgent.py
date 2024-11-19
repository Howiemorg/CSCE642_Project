import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from torch.optim import Adam
import os
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

class ActorNetwork(nn.Module):
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

        # Final layers for mean (mu) and standard deviation (std) of actions
        self.fc_mu = nn.Linear(hidden_size[1]*4, act_dim, dtype=torch.float32)
        self.fc_log_std = nn.Linear(hidden_size[1]*4, act_dim, dtype=torch.float32)        


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
            if layer == self.fc_mu or layer == self.fc_log_std:
                # Initialize the output layers with smaller variance
                nn.init.uniform_(layer.weight, -3e-3, 3e-3)
                nn.init.constant_(layer.bias, 0)
            else:
                # Use Xavier initialization for non-output FC layers
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

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

        


        # # x_deltas = self.delta_fc(x_deltas)
        # print(x_deltas.shape)
        # print(x_image.shape)
        # print(x_position.shape)

        # Concatenate image and position features
        x_combined = torch.cat([x_image, x_position, x_deltas, x_memory], dim=-1)
        
        # Concatenate image features and position features
        # x_combined = torch.cat((x_image, x_position), dim=1)
        
        # Output mean (mu) and log_std for each action dimension
        mu = self.fc_mu(x_combined)
        log_std = self.fc_log_std(x_combined)

        # # Separate the last element of mu and apply ReLU to make it positive
        # mu_last_positive = F.relu(mu[:, -1:])
        # mu_rest = mu[:, :-1]
        
        # # Concatenate back to form the final mu with the last element positive
        # mu = torch.cat((mu_rest, mu_last_positive), dim=1)
        
        # Apply softplus to log_std to ensure std is positive
        std = F.softplus(log_std) # or use softplus: F.softplus(log_std)

        
        # Squash the action values to the bounds of the action space
        mu = self.action_low + 0.5 * (self.action_high - self.action_low) * (torch.tanh(mu) + 1)
        # std = self.action_low + 0.5 * (self.action_high - self.action_low) * (torch.tanh(std) + 1)
        # std = torch.clamp(std, min=1e-6)
        # return mu
        
        return mu, std


class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_size, replay_size):
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
        
        # Final output layer to predict the value, combining CNN (256) and position (64) outputs
        self.fc_value = nn.Linear(hidden_size[1]*4, 1, dtype=torch.float32)

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

    def forward(self, x):
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


        # Concatenate image and position features
        x_combined = torch.cat([x_image, x_position, x_deltas, x_memory], dim=-1)
        
        # Output the value prediction
        value = self.fc_value(x_combined)
        return torch.squeeze(value, -1)

class A2CAgent(AbstractSolver):
    def __init__(self, env, eval_env, options):
        super().__init__(env, eval_env, options)
        # Create actor-critic network
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.actionBuf = MDPBuffer(self.options.replay_memory_size, env.observation_space_shape+3)


        self.actor = ActorNetwork(
            env.observation_space_shape, env.action_space.shape[0], env.bounds, self.options.layers, self.actionBuf.state_dim* self.options.replay_memory_size#, self.options.layers
        ).to(self.device)
        self.policy = self.create_greedy_policy()

        self.actor_optimizer = Adam(
            self.actor.parameters(), lr=self.options.actor_alpha)

        self.critic = CriticNetwork(
            env.observation_space_shape, self.options.layers, self.actionBuf.state_dim* self.options.replay_memory_size#, , self.options.layers
        ).to(self.device)

        self.critic_optimizer = Adam(
            self.critic.parameters(), lr=self.options.critic_alpha) 
        
        print((env.observation_space_shape, 3))

        self.env = env

    def create_greedy_policy(self):
        """
        Creates a greedy policy.


        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """

        def policy_fn(state):
            state_tensor = self.preprocess_state(state)

            mus, _ = self.actor(state_tensor)
            mus = mus.squeeze(0)
            return mus.detach().cpu().numpy()

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
        attitude = state["attitude"][:-11]
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


    def select_action(self, state):
        """
        Selects an action given state.

        Returns:
            The selected action (as an int)
            The probability of the selected action (as a tensor)
            The critic's value estimate (as a tensor)
        """
        # print(state)
        state_tensor = self.preprocess_state(state)
        # print(state_tensor)
        mus, stds = self.actor(state_tensor)
        value = self.critic(state_tensor)

        if torch.isnan(mus).all():
            print("attitude", state["attitude"])
            print("stds:",stds.squeeze(0))
            print("mus:",mus.squeeze(0))
        mus = mus.squeeze(0)
        stds = stds.squeeze(0) + 1e-8
        print("target_deltas", state["target_deltas"])

        normal = Normal(mus, stds)
        sample = normal.sample()
        # print("SAMPLES:", sample)
        log_prob = normal.log_prob(sample).sum()
        # print("prob:",log_prob)

        return mus, log_prob, value

    def update_actor_critic(self, advantage, prob, estimate, value):
        """
        Performs actor critic update.

        args:
            advantage: Advantage of the chosen action (tensor).
            prob: Probability associated with the chosen action (tensor).
            value: Critic's state value estimate (tensor).
        """
        # Compute loss
        # print("Before Loss Update")
        # print("Advantage:", advantage)
        # print("Prob:", prob)
        actor_loss = self.actor_loss(advantage.detach(), prob)
        # print("Actor Loss:", actor_loss)
        critic_loss = self.critic_loss(estimate, value).mean()
        # print("critic_loss Loss:", critic_loss)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor critic

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
        torch.save(self.actor.state_dict(), actor_dir)
        torch.save(self.critic.state_dict(), critic_dir)

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
        self.actor.load_state_dict(torch.load(actor_dir))
        self.actor.eval()
        self.critic.load_state_dict(torch.load(critic_dir))
        self.critic.eval()




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
        # print("STATE RESET:", torch.tensor(state["rgba_cam"]).shape)
        # i =0
        for _ in range(self.options.steps):
            # print(f"State {i}")
            # i+=1
            action, action_prob, estimate = self.select_action(state)
            # print("action", action)
            next_state, reward, done, _ = self.step(action.cpu().detach().numpy())
            next_state_tensor = self.preprocess_state(next_state)

            # advantage = reward - estimate
            next_state_estimate  = self.critic(next_state_tensor)
            # print(f"State value {value}")
            advantage = (reward + (1-done)*self.options.gamma * next_state_estimate) - estimate
            self.update_actor_critic(advantage, action_prob, estimate, advantage+estimate)
           
            state = next_state
            if (done):
                # self.update_actor_critic(advantage, action_prob, estimate)
                break
            # print(next_state)

    def actor_loss(self, advantage, log_prob):
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
        loss = -advantage * log_prob
        loss -= torch.exp(log_prob)*log_prob *0.01# add entropy to encourage exploration
        return  loss
        # return -advantage * torch.log(prob)

    def critic_loss(self, estimate, value):
        """
        The integral of the critic gradient

        args:
            advantage: Advantage of the chosen action.
            value: Critic's state value estimate.

        Returns:
            The unreduced loss (as a tensor).
        """
        # return torch.tensor(advantage.pow(2), dtype=torch.float32)
        return F.mse_loss(estimate, value)
    def __str__(self):
        return "A2CAgent"

    # def plot(self, stats, smoothing_window=20, final=False):
    #     plotting.plot_episode_stats(stats, smoothing_window, final=final)
