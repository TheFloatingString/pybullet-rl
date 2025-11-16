import pybullet as p
import pybullet_data
import numpy as np
import imageio_ffmpeg
import tqdm
import gymnasium as gym
import sympy
from gymnasium import spaces
from math import pi, sin, cos
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class RewardLoggerCallback(BaseCallback):
    """Custom callback to record rewards during training"""
    def __init__(self):
        super().__init__()
        self.rewards = []

    def _on_step(self) -> bool:
        # Record the reward from this step
        if len(self.locals.get('rewards', [])) > 0:
            self.rewards.append(np.mean(self.locals['rewards']))
        return True


class RobotSimulation(gym.Env):
    """Custom Gymnasium environment for Kuka arm simulation with PyBullet"""
    
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 30}
    
    def __init__(self, video_path='simulation.mp4', record_video=True):
        """Initialize PyBullet simulation with video recording"""
        super().__init__()
        
        # Connect to PyBullet
        p.connect(p.DIRECT)  # Use DIRECT for headless, GUI for visualization
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        
        # Load environment
        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF(
            "table/table.urdf", 
            basePosition=[1.0, -0.2, 0.0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, pi/4])
        )
        
        # Load robot
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            basePosition=[1.4, -0.2, 0.6],
            useFixedBase=True
        )
        
        # Camera parameters
        self.cam_width = 640
        self.cam_height = 480
        self.cam_target = [0.95, -0.2, 0.8]
        self.cam_distance = 2.5
        self.cam_yaw = -50
        self.cam_pitch = -35
        self.cam_roll = 0
        self.cam_fov = 60
        
        # Video recording setup
        self.record_video = record_video
        self.vid_writer = None
        if self.record_video:
            self.vid_writer = imageio_ffmpeg.write_frames(
                video_path,
                (self.cam_width, self.cam_height),
                fps=30
            )
            self.vid_writer.send(None)  # Initialize generator
        
        # Simulation state
        self.timestep = 0
        self.num_joints = p.getNumJoints(self.robot_id)
        self.end_effector_index = 6  # Last link of Kuka arm
        
        # Define Gymnasium spaces
        # Observation: joint positions + joint velocities
        obs_dim = self.num_joints * 2  # positions + velocities
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # Action: target joint positions (normalized)
        self.action_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(self.num_joints,), 
            dtype=np.float32
        )
        
        # Target end effector position (goal to reach)
        self.target_pos = np.array([0.85, 1, 1.2])

    def _get_observation(self):
        """Get current joint positions and velocities as observation"""
        joint_states = p.getJointStates(self.robot_id, range(self.num_joints))
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        
        # Concatenate positions and velocities
        observation = np.array(joint_positions + joint_velocities, dtype=np.float32)
        return observation
    
    def _get_end_effector_pos(self):
        """Get current end effector position"""
        link_state = p.getLinkState(self.robot_id, self.end_effector_index)
        return np.array(link_state[0])  # World position
    
    def _compute_reward(self):
        """Compute reward based on end effector proximity to target"""
        ee_pos = self._get_end_effector_pos()
        distance = np.linalg.norm(ee_pos - self.target_pos)
        
        # Reward is negative distance (closer = higher reward)
        reward = -distance
        
        # Bonus for being very close to target
        if distance < 0.1:
            reward += 5.0
        
        # Small penalty for high joint velocities (encourage smooth motion)
        joint_states = p.getJointStates(self.robot_id, range(self.num_joints))
        joint_velocities = [state[1] for state in joint_states]
        velocity_penalty = -0.01 * sum(v**2 for v in joint_velocities)
        
        return reward + velocity_penalty
    
    def reset(self, seed=None, options=None):
        """Reset robot to initial configuration"""
        super().reset(seed=seed)
        
        # Home position for Kuka arm
        initial_positions = [0, -0.5, 0, 1.8, 0, -1.2, 0]
        
        for joint_idx in range(self.num_joints):
            p.resetJointState(self.robot_id, joint_idx, initial_positions[joint_idx])
            # Set motors with proper force to hold position
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=initial_positions[joint_idx],
                force=500
            )
        
        self.timestep = 0
        observation = self._get_observation()
        info = {'end_effector_pos': self._get_end_effector_pos()}
        
        return observation, info
    
    def step(self, action):
        """
        Execute one simulation step
        
        Args:
            action: Target joint positions (normalized to [-1, 1])
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Scale action from [-1, 1] to actual joint limits
        # For Kuka, using reasonable joint limits
        joint_limits = np.array([2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.05])
        joint_targets = action * joint_limits
        
        # Apply motor controls
        for joint_idx in range(self.num_joints):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=joint_targets[joint_idx],
                force=500,
                maxVelocity=1.0
            )
        
        # Step physics simulation
        p.stepSimulation()
        
        # Record video frame (at 30fps, PyBullet runs at 240Hz by default)
        if self.record_video and self.timestep % 8 == 0:
            self._capture_frame()
        
        self.timestep += 1
        
        # Get observation
        observation = self._get_observation()
        
        # Compute reward
        reward = self._compute_reward()
        
        # Check if episode is done
        ee_pos = self._get_end_effector_pos()
        distance_to_target = np.linalg.norm(ee_pos - self.target_pos)
        terminated = distance_to_target < 0.05  # Success condition
        truncated = self.timestep >= 1000  # Time limit
        
        # Additional info
        info = {
            'timestep': self.timestep,
            'end_effector_pos': ee_pos,
            'distance_to_target': distance_to_target
        }
        
        return observation, reward, terminated, truncated, info
    
    def _capture_frame(self):
        """Capture and write a video frame"""
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            self.cam_target,
            self.cam_distance,
            self.cam_yaw,
            self.cam_pitch,
            self.cam_roll,
            upAxisIndex=2
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            self.cam_fov,
            self.cam_width / self.cam_height,
            0.01,
            100
        )
        
        # Get camera image
        img = p.getCameraImage(
            self.cam_width,
            self.cam_height,
            view_matrix,
            proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Extract RGB data (ignore alpha channel)
        rgb_array = np.array(img[2], dtype=np.uint8).reshape(
            self.cam_height, self.cam_width, 4
        )[:, :, :3]
        
        # Send frame to video writer
        self.vid_writer.send(np.ascontiguousarray(rgb_array))
    
    def render(self):
        """Render the environment (for compatibility)"""
        if self.record_video:
            self._capture_frame()
    
    def close(self):
        """Clean up simulation and finalize video"""
        if self.vid_writer is not None:
            self.vid_writer.close()
            print("Video saved successfully!")
        
        p.disconnect()
        print("Simulation closed.")


def main():
    """Run a sample RL training"""
    print("Starting Kuka arm simulation with RL training...")
    
    # Create simulation environment
    env = RobotSimulation(video_path='robot_demo.mp4', record_video=True)
    
    # Create callback
    callback = RewardLoggerCallback()
    
    # Create PPO model
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        n_steps=128,
        batch_size=64,
        learning_rate=3e-4,
        n_epochs=10
    )
    
    # Train the model
    print("Training model...")
    model.learn(total_timesteps=2000, callback=callback)
    
    # Test the trained model
    print("\nTesting trained model...")
    obs, info = env.reset()
    for i in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode finished at step {i}")
            obs, info = env.reset()
    
    # Close and save
    env.close()
    print("Done! Check 'robot_demo.mp4'")
    
    # Print reward statistics
    if callback.rewards:
        print(f"\nTraining statistics:")
        print(f"  Mean reward: {np.mean(callback.rewards):.3f}")
        print(f"  Max reward: {np.max(callback.rewards):.3f}")
        print(f"  Min reward: {np.min(callback.rewards):.3f}")


if __name__ == "__main__":
    main()