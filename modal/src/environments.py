from typing import Any, Dict, List, Tuple

import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
import wandb
from gymnasium import spaces
from numpy.typing import NDArray
from stable_baselines3.common.callbacks import BaseCallback


def _setup_pybullet_physics() -> None:
    """Initialize PyBullet connection and physics settings."""
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)


def _load_environment_plane() -> int:
    """Load the ground plane into the simulation."""
    return p.loadURDF("plane.urdf")


def _load_kuka_robot() -> int:
    """Load the Kuka IIWA robot into the simulation."""
    return p.loadURDF(
        "kuka_iiwa/model.urdf",
        basePosition=[0, 0, 0],
        useFixedBase=True,
    )


def _create_observation_space(num_joints: int) -> spaces.Box:
    """Create observation space for joint positions and velocities."""
    return spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(num_joints * 2 + 3,),
        dtype=np.float32,
    )


def _create_action_space(num_joints: int) -> spaces.Box:
    """Create action space for joint velocities."""
    return spaces.Box(
        low=-20,
        high=20,
        shape=(num_joints,),
        dtype=np.float32,
    )


def _generate_random_cylindrical_position() -> NDArray[np.float32]:
    """Generate random 3D position using cylindrical coordinates."""
    z = 0
    r = np.random.uniform(low=1.5, high=2)
    theta = np.random.uniform(low=0, high=2 * np.pi)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.array([x, y, z])


def _generate_random_joint_pose(num_joints: int) -> NDArray[np.float32]:
    """Generate random initial joint positions for the robot."""
    # Reference positions: [0, -0.5, 0, 1.8, 0, -1.2, 0]
    positions = np.zeros(num_joints)
    positions[0] = np.random.uniform(low=-np.pi, high=np.pi)
    positions[1] = np.random.uniform(low=-0.6, high=-0.4)
    positions[2] = np.random.uniform(low=-np.pi / 4, high=np.pi / 4)
    positions[3] = np.random.uniform(low=1.7, high=1.9)
    positions[4] = np.random.uniform(low=-np.pi / 4, high=np.pi / 4)
    positions[5] = np.random.uniform(low=-1.5, high=-1.1)
    positions[6] = np.random.uniform(low=-np.pi / 4, high=np.pi / 4)
    return positions


def _set_joint_position(robot_id: int, joint_idx: int, position: float) -> None:
    """Set a joint to a specific position with motor control."""
    p.resetJointState(robot_id, joint_idx, position)
    p.setJointMotorControl2(
        robot_id,
        joint_idx,
        p.POSITION_CONTROL,
        targetPosition=position,
        force=500,
    )


def _set_joint_velocity(robot_id: int, joint_idx: int, velocity: float) -> None:
    """Set a joint to a specific velocity with motor control."""
    p.setJointMotorControl2(
        robot_id,
        joint_idx,
        p.VELOCITY_CONTROL,
        targetVelocity=velocity,
        force=500,
    )


def _get_joint_states(robot_id: int, num_joints: int) -> NDArray[np.float32]:
    """Get current joint positions and velocities."""
    joint_states = [p.getJointState(robot_id, i) for i in range(num_joints)]
    positions = [state[0] for state in joint_states]
    velocities = [state[1] for state in joint_states]
    return np.array(positions + velocities, dtype=np.float32)


def _get_link_position(robot_id: int, link_idx: int) -> List[float]:
    """Get the world position of a specific link."""
    state = p.getLinkState(robot_id, link_idx)
    return list(state[0])


def _compute_distance_reward(
    current_pos: List[float], target_pos: NDArray[np.float32]
) -> float:
    """Compute negative distance reward between positions."""
    return -np.linalg.norm(target_pos - np.array(current_pos))


def _render_camera_view() -> NDArray[np.uint8]:
    """Render the simulation from a fixed camera viewpoint."""
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, 0],
        distance=2.0,
        yaw=45,
        pitch=-30,
        roll=0,
        upAxisIndex=2,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=1.0, nearVal=0.1, farVal=100.0
    )
    img = p.getCameraImage(
        640,
        480,
        view_matrix,
        proj_matrix,
        renderer=p.ER_TINY_RENDERER,
    )
    return img[2][:, :, :3]


class RewardLoggerCallback(BaseCallback):
    """Callback to log episode rewards during training."""

    def __init__(self) -> None:
        super().__init__()
        self.rewards: List[float] = []

    def _on_step(self) -> bool:
        """Log rewards at each step."""
        if len(self.locals.get("rewards", [])) > 0:
            reward = np.mean(self.locals["rewards"])
            self.rewards.append(reward)
            wandb.log({"reward": reward, "timestep": self.n_calls})
        return True


class KukaEnv(gym.Env):
    """
    Gymnasium environment for Kuka IIWA robot arm.

    observation space:
        np.NDArray of size 17 for
            joint positions (7),
            joint velocities (7)
            target ee position (3)
    action space: np.NDArray of size 7 for joint velocities in rad/s
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.render_mode = render_mode

        _setup_pybullet_physics()
        self.plane_id: int = _load_environment_plane()
        self.robot_id: int = _load_kuka_robot()

        self.num_joints: int = p.getNumJoints(self.robot_id)
        self.timestep: int = 0
        self.max_steps: int = 1000

        self.observation_space = _create_observation_space(self.num_joints)
        self.action_space = _create_action_space(self.num_joints)
        self.target_effector_position: NDArray[np.float32] | None = None
        self.video_logging_id: int | None = None

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[NDArray[np.float32], Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        initial_positions = _generate_random_joint_pose(self.num_joints)
        self.target_effector_position = _generate_random_cylindrical_position()

        for joint_idx in range(self.num_joints):
            _set_joint_position(
                self.robot_id,
                joint_idx,
                initial_positions[joint_idx],
            )

        self.timestep = 0
        joint_states = _get_joint_states(self.robot_id, self.num_joints)
        observation = np.concatenate((joint_states, self.target_effector_position))
        info = {"end_effector_pos": self._get_end_effector_pos()}

        return observation, info

    def step(
        self, action: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        """Execute one step in environment."""
        for joint_idx in range(self.num_joints):
            _set_joint_velocity(self.robot_id, joint_idx, action[joint_idx])

        p.stepSimulation()
        self.timestep += 1

        joint_states = _get_joint_states(self.robot_id, self.num_joints)
        observation = np.concatenate((joint_states, self.target_effector_position))
        reward = self._compute_reward()
        terminated = False
        truncated = self.timestep >= self.max_steps
        info = {"end_effector_pos": self._get_end_effector_pos()}

        return observation, reward, terminated, truncated, info

    def _get_end_effector_pos(self) -> List[float]:
        """Get end effector position in xyz world coordinates."""
        return _get_link_position(self.robot_id, self.num_joints - 1)

    def _compute_reward(self) -> float:
        """Compute reward for current state."""
        curr_pos = self._get_end_effector_pos()
        return _compute_distance_reward(curr_pos, self.target_effector_position)

    def render(self) -> NDArray[np.uint8] | None:
        """Render environment state."""
        if self.render_mode != "rgb_array":
            return None
        return _render_camera_view()

    def close(self) -> None:
        """Clean up environment resources."""
        p.disconnect()
