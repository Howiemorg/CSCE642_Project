# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) inform Guni Sharon at 
# guni@tamu.edu regarding your usage (relevant statistics is reported to NSF).
# The development of this assignment was supported by NSF (IIS-2238979).
# Contributors:
# The core code base was developed by Guni Sharon (guni@tamu.edu).

from abc import ABC, abstractmethod
from enum import Enum
import gymnasium as gym
import time
from utils import plotting


class AbstractSolver(ABC):
    def __init__(self, env, eval_env, options):
        self.statistics = [0] * len(Statistics)
        self.env = env
        self.eval_env = eval_env
        self.options = options
        self.total_steps = 0
        self.render = False

    def init_stats(self):
        self.statistics[1:] = [0] * (len(Statistics) - 1)

    def step(self, action):
        """
        Take one step in the environment while keeping track of statistical information
        Param:
            action:
        Return:
            next_state: The next state
            reward: Immediate reward
            done: Is next_state terminal
            info: Gym transition information
        """
        # print(action)
        next_state, reward, terminated, truncated, info = self.env.step(action)

        # Update statistics
        self.statistics[Statistics.Rewards.value] += reward
        self.statistics[Statistics.Steps.value] += 1
        self.total_steps += 1

        return next_state, reward, terminated or truncated, info

    def run_greedy(self):
        """
        Run the greedy policy.
        """
        policy = self.create_greedy_policy()
        state, _ = self.eval_env.reset()

        rewards = 0
        steps = 0
        for _ in range(self.options.steps):
            action = policy(state)
            state, reward, done, _, _ = self.eval_env.step(action)

            rewards += reward
            steps += 1
            if done:
                break
        return rewards, steps

    def close(self):
        pass

    @abstractmethod
    def train_episode(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def create_greedy_policy(self):
        pass

    @staticmethod
    def get_out_header():
        ans = "Domain,Solver"
        for s in Statistics:
            ans += "," + s.name
        return ans

    def plot(self, stats, smoothing_window=20, final=False):
        plotting.plot_episode_stats(stats, smoothing_window, final=final)

    def get_stat(self):
        try:
            domain = self.env.unwrapped.spec.id
        except:
            domain = self.env.name
        ans = "{},{}".format(domain, str(self))
        for s in Statistics:
            ans += "," + str(self.statistics[s.value])
        return ans


class Statistics(Enum):
    Episode = 0
    Rewards = 1
    Steps = 2
