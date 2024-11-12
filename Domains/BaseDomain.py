from PyFlyt.gym_envs.quadx_envs.quadx_waypoints_env import QuadXWaypointsEnv
# from PyFlyt.models import race_gate
from typing import Any, Literal
import numpy as np
import pybullet as p
from gymnasium import spaces
import os
from PyFlyt.core import loadOBJ, obj_collision, obj_visual

class BaseDomain(QuadXWaypointsEnv):
    """BaseDomain: Quad Waypoints Environment.

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
        camera_resolution: tuple[int, int] = (128, 128),
        num_targets: int = 1,
        **kwargs
    ):
    
        """Initialize BaseDomain with optional duck parameters.

        Args:
            duck_position (tuple): Position where the duck is loaded.
            kwargs: Additional arguments passed to the QuadXGatesEnv.
        """
        # Call the base class initializer
        kwargs["num_targets"]=num_targets
        super().__init__(**kwargs)
        
        self.camera_resolution = camera_resolution
        self.num_targets =num_targets

        self.duck_position = duck_position

        self.observation_space["rgba_cam"]= spaces.Box(
                    low=0.0, high=255.0, shape=(4, *camera_resolution), dtype=np.uint8
                )


        # print(self.observation_space["attitude"].shape[0])
        # print(self.observation_space["target_deltas"])
        self.observation_space_shape = (self.observation_space["attitude"].shape[0] 
                                        # + (self.observation_space["rgba_cam"].shape[0]*self.observation_space["rgba_cam"].shape[1]*self.observation_space["rgba_cam"].shape[2]) #only a single frame
                                        + (self.num_targets*3)) # 3 is for 3-dimensions of the deltas
        # print(self.observation_space_shape)

        # self.load_duck() 
        

    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[dict[Literal["attitude", "target_deltas", "rgba_cam"], np.ndarray], dict]:
        """Resets the environment.

        Args:
            seed: seed to pass to the base environment.
            options: None

        """
        aviary_options = dict()
        aviary_options["use_camera"] = True
        aviary_options["use_gimbal"] = False
        aviary_options["camera_resolution"] = self.camera_resolution
        aviary_options["camera_angle_degrees"] = 15.0

        super().begin_reset(seed, options, aviary_options)
        self.waypoints.reset(self.env, self.np_random)
        self.info["num_targets_reached"] = 0
        super().end_reset()

        return self.state, self.info

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
        """This returns the observation as well as the distances to target.

        - "attitude" (Box)
            - ang_vel (vector of 3 values)
            - ang_pos (vector of 3/4 values)
            - lin_vel (vector of 3 values)
            - lin_pos (vector of 3 values)
            - previous_action (vector of 4 values)
            - auxiliary information (vector of 4 values)
        - "target_deltas" (Graph)
            - list of body_frame distances to target (vector of 3/4 values)
        """
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = super().compute_attitude()
        aux_state = super().compute_auxiliary()
        # print(aux_state)

        # rotation matrix
        # rotation = np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3).T

        # drone to target
        target_deltas =self.waypoints.distance_to_targets(
            ang_pos, lin_pos, quaternion
        )
        self.dis_error_scalar = np.linalg.norm(target_deltas[0])

        # combine everything
        new_state: dict[
            Literal["attitude", "rgba_cam", "target_deltas"], np.ndarray
        ] = dict()
        if self.angle_representation == 0:
            new_state["attitude"] = np.concatenate(
                [ang_vel, ang_pos, lin_vel, lin_pos, self.action, aux_state], axis=-1
            )
        elif self.angle_representation == 1:
            new_state["attitude"] = np.concatenate(
                [ang_vel, quaternion, lin_vel, lin_pos, self.action, aux_state], axis=-1
            )

        # grab the image
        img = self.env.drones[0].rgbaImg.astype(np.uint8)
        new_state["rgba_cam"] = np.moveaxis(img, -1, 0)

        # distances to targets
        new_state["target_deltas"] = target_deltas

        self.state: dict[
            Literal["attitude", "rgba_cam", "target_deltas"], np.ndarray
        ] = new_state

    def compute_term_trunc_reward(self):
        """Compute termination, truncation, and modified reward function."""
        # Call the base class version if you want to use it as a starting point
        super().compute_term_trunc_reward()

        # Modify the reward calculation here
        if self.waypoints.target_reached:
            self.reward += 200.0  # Example change to reward
            # Add any additional reward conditions specific to the duck, if needed
