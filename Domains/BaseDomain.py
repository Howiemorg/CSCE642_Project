from PyFlyt.gym_envs.quadx_envs.quadx_gates_env import QuadXGatesEnv
import os
from PyFlyt.core import loadOBJ, obj_collision, obj_visual

class BaseDomain(QuadXGatesEnv):
    """Base Domain Environment.

    This environment inherits from QuadXGatesEnv but includes additional
    features such as loading a duck and a modified reward function.
    """

    def __init__(
        self,
        duck_position: tuple[float, float, float] = (1.0, 1.0, 0.5),
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
        self.load_duck() 
        # Add any other specific initialization you need here

    def reset(self, *args, **kwargs):
        """Resets the environment, including loading the duck."""
        # Optionally, modify the reset process or call super() if the base class reset is sufficient
        obs, info = super().reset(*args, **kwargs)
        self.load_duck()  # Load the duck as an additional step after reset
        return obs, info

    def load_duck(self):
        """Loads a duck object into the environment."""
        duck_obj_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../Models/duck.obj"
        )
        visualId = obj_visual(self.env, duck_obj_dir)
        collisionId = obj_collision(self.env, duck_obj_dir)
        loadOBJ(
            self.env,
            visualId=visualId,
            collisionId=collisionId,
            baseMass=1.0,
            basePosition=self.duck_position,
        )

        self.env.register_all_new_bodies()

    def compute_state(self) -> None:
        super().compute_state()
        raise NotImplementedError

    def compute_term_trunc_reward(self):
        """Compute termination, truncation, and modified reward function."""
        # Call the base class version if you want to use it as a starting point
        super().compute_term_trunc_reward()

        # Modify the reward calculation here
        if self.target_reached:
            self.reward += 200.0  # Example change to reward
            # Add any additional reward conditions specific to the duck, if needed
