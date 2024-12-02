# Plotting code by by Denny Britz
# repository: https://github.com/dennybritz/reinforcement-learning

import numpy as np
import pandas as pd
from collections import namedtuple
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

matplotlib.style.use("ggplot")
from mpl_toolkits.mplot3d import Axes3D

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])

def plot_episode_stats(stats, smoothing_window=20, final=False):
    assert stats.episode_lengths[0] >= 0, "Can't print DP statistics"

    # Plot the episode reward over time
    fig = plt.figure(1)
    rewards = pd.Series(stats.episode_rewards)
    rewards_smoothed = rewards.rolling(
        smoothing_window, min_periods=smoothing_window
    ).mean()
    plt.clf()
    if final:
        plt.title("Result")
    else:
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Episodic Return")
    plt.plot(rewards, label="Raw", c="b", alpha=0.3)
    if len(rewards_smoothed) >= smoothing_window:
        plt.plot(
            rewards_smoothed,
            label=f"Smooth (win={smoothing_window})",
            c="k",
            alpha=0.7,
        )
    plt.legend()
    if final:
        # plt.pause(5)
        plt.show(block=True)
    else:
        plt.pause(0.1)

# df = pd.read_csv("./Results/out.csv")
# print(df.columns)
# print(df[["Algorithm"]])
# ddpg_training = df[df["Algorithm"] == "DDPG"]
# targets_reached = (ddpg_training["Reward"] > 0).sum(axis=None)
# print(targets_reached)

# targets_reached_df = ddpg_training[ddpg_training["Reward"] > 0]["Steps"].min(axis=None)
# print(targets_reached_df)
# # a2c_training = df[df["Algorithm"] == "A2CEligibility"]
# print(ddpg_training)
# # stats = EpisodeStats(episode_rewards=ddpg_training["Reward"], episode_lengths=ddpg_training["Steps"])
# plot_episode_stats(stats, 5000, final=True)
# print(a2c_training)
# stats = EpisodeStats(episode_rewards=a2c_training["Reward"], episode_lengths=[12, 12])
# plot_episode_stats(stats, 5000, final=True)