import pybullet as p
import pybullet_data
import numpy as np
import imageio_ffmpeg
import tqdm
from math import pi, sin, cos


class RobotSimulation:
    def __init__(self, video_path='simulation.mp4', record_video=True):
        """Initialize PyBullet simulation with video recording"""
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
        self.cube_id = p.loadURDF(
            "cube.urdf",
            basePosition=[0.85, -0.2, 0.65],
            globalScaling=0.05
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
        self.cam_target = [0.95, -0.2, 0.2]
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
        
        # Reset to initial pose
        self.reset()
    
    def reset(self):
        """Reset robot to initial configuration"""
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
                force=500  # Sufficient force to hold against gravity
            )
        
        self.timestep = 0
        return self._get_observation()
    
    def step(self):
        """Execute one simulation step with animated robot motion"""
        # Animate the robot - simple sine wave motion for demonstration
        t = self.timestep * 0.01
        
        # Move joints in a coordinated pattern
        joint_targets = [
            0.3 * sin(t),           # Joint 0
            -0.5 + 0.2 * sin(t),    # Joint 1
            0.2 * cos(t),           # Joint 2
            1.8 + 0.3 * sin(2*t),   # Joint 3
            0.3 * sin(t),           # Joint 4
            -1.2 + 0.2 * cos(t),    # Joint 5
            0.4 * sin(3*t)          # Joint 6
        ]
        
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
        return self._get_observation()
    
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
    
    def _get_observation(self):
        """Get current state observation"""
        joint_states = p.getJointStates(self.robot_id, range(self.num_joints))
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        
        cube_pos, cube_orn = p.getBasePositionAndOrientation(self.cube_id)
        
        return {
            'joint_positions': joint_positions,
            'joint_velocities': joint_velocities,
            'cube_position': cube_pos,
            'cube_orientation': cube_orn
        }
    
    def close(self):
        """Clean up simulation and finalize video"""
        if self.vid_writer is not None:
            self.vid_writer.close()  # CRITICAL: Finalize video file
            print("Video saved successfully!")
        
        p.disconnect()
        print("Simulation closed.")


def main():
    """Run a sample simulation"""
    print("Starting robot simulation...")
    
    # Create simulation
    env = RobotSimulation(video_path='robot_demo.mp4', record_video=True)
    
    # Run simulation for 10 seconds (30fps = 300 frames, 240Hz physics = 2400 steps)
    num_steps = 2400
    
    print(f"Running {num_steps} simulation steps...")
    for i in tqdm.trange(num_steps):
        env.step()
    
    # IMPORTANT: Close to finalize video
    env.close()
    print("Done! Check 'robot_demo.mp4'")


if __name__ == "__main__":
    main()
