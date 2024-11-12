from PyFlyt.gym_envs.quadx_envs.quadx_gates_env import QuadXGatesEnv
# from PyFlyt.models import race_gate
from typing import Any, Literal
import numpy as np
import pybullet as p
from gymnasium import spaces
import os
from PyFlyt.core import loadOBJ, obj_collision, obj_visual

class BaseDomain(QuadXGatesEnv):
    """BaseDomain: Quad Gates Environment.

    Actions are vp, vq, vr, T, ie: angular rates and thrust

    The target is a set of `[x, y, z, yaw]` targets in space

    Reward is -(distance from waypoint + angle error) for each timestep, and -100.0 for hitting the ground.

    Args:
        flight_mode (int): the flight mode of the UAV
        num_targets (int): num_targets
        goal_reach_distance (float): goal_reach_distance
        min_gate_height (float): min_gate_height
        max_gate_angles (list[float]): max_gate_angles
        min_gate_distance (float): min_gate_distance
        max_gate_distance (float): max_gate_distance
        camera_resolution (tuple[int, int]): camera_resolution
        max_duration_seconds (float): max_duration_seconds
        angle_representation (Literal["euler", "quaternion"]): can be "euler" or "quaternion".
        agent_hz (int): looprate of the agent to environment interaction.
        render_mode (None | Literal["human", "rgb_array"]): render_mode
        render_resolution (tuple[int, int]): render_resolution

    """

   
    def __init__(
        self,
        duck_position: tuple[float, float, float] = (1.0, 1.0, 0.5),
        num_targets: int = 2,
        **kwargs
    ):
        """Initialize BaseDomain with optional duck parameters.

        Args:
            duck_position (tuple): Position where the duck is loaded.
            kwargs: Additional arguments passed to the QuadXGatesEnv.
        """
        # Call the base class initializer
        super().__init__(**kwargs)
        

        self.duck_position = duck_position


        self.observation_space_shape = (self.observation_space["attitude"].shape[0] 
                                        # + (self.observation_space["rgba_cam"].shape[0]*self.observation_space["rgba_cam"].shape[1]*self.observation_space["rgba_cam"].shape[2]) #only a single frame
                                        + (self.num_targets*3)) # 3 is for 3-dimensions of the deltas
        # print(self.observation_space_shape)

        # self.load_duck() 

    def reset(self, *args, **kwargs):
        """Resets the environment, including loading the duck."""
        # Optionally, modify the reset process or call super() if the base class reset is sufficient
        obs, info = super().reset(*args, **kwargs)
        # self.load_duck()  # Load the duck as an additional step after reset
        return obs, info

    # def load_duck(self):
    #     """Loads a duck object into the environment."""
    #     duck_obj_dir = os.path.join(
    #         os.path.dirname(os.path.realpath(__file__)),
    #         "../Models/duck.obj"
    #     )
    #     visualId = obj_visual(self.env, duck_obj_dir)
    #     collisionId = obj_collision(self.env, duck_obj_dir)
    #     loadOBJ(
    #         self.env,
    #         visualId=visualId,
    #         collisionId=collisionId,
    #         baseMass=1.0,
    #         basePosition=self.duck_position,
    #     )

    #     self.env.register_all_new_bodies()

    def compute_state(self) -> None:
        super().compute_state()
        # raise NotImplementedError

    def compute_term_trunc_reward(self):
        """Compute termination, truncation, and modified reward function."""
        # Call the base class version if you want to use it as a starting point
        super().compute_term_trunc_reward()

        # Modify the reward calculation here
        if self.target_reached:
            self.reward += 200.0  # Example change to reward
            # Add any additional reward conditions specific to the duck, if needed
