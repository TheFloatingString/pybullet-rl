import modal
import os
import wandb


app = modal.App("pybullet-rl")

# Create image with pybullet, torch, and stable-baselines3
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pybullet",
        "torch",
        "stable-baselines3",
        "wandb",
    )
    .pip_install("imageio", "imageio-ffmpeg")
)

# Create mount for the src directory
src_mount = modal.Mount.from_local_dir("./src", remote_path="/root/src")


@app.function(
    image=image,
    gpu="T4",  # Use NVIDIA T4 GPU
    timeout=3600,  # 1 hour timeout
    mounts=[src_mount],
)
def run_pybullet(wandb_api_key: str) -> None:
    """Run PyBullet on GPU instance with torch and stable-baselines3."""
    import os

    import pybullet as p
    import pybullet_data
    import torch
    import wandb
    from math import pi
    from src.create_objects import create_rgb_axes

    # Set wandb API key as environment variable
    os.environ["WANDB_API_KEY"] = wandb_api_key

    # Login to wandb with API key (relogin=True forces re-authentication)
    wandb.login(key=wandb_api_key, relogin=True)

    # Initialize wandb (entity is the team/username)
    wandb.init(entity="larryl729-team", project="pybullet-rl", name="modal-run")

    print("PyBullet GPU Instance Started!")
    pybullet_version = p.getAPIVersion()
    pytorch_version = torch.__version__
    cuda_available = torch.cuda.is_available()

    print(f"PyBullet version: {pybullet_version}")
    print(f"PyTorch version: {pytorch_version}")
    print(f"CUDA available: {cuda_available}")

    # Log environment configuration
    config = {
        "pybullet_version": pybullet_version,
        "pytorch_version": pytorch_version,
        "cuda_available": cuda_available,
    }

    if torch.cuda.is_available():
        cuda_device = torch.cuda.get_device_name(0)
        print(f"CUDA device: {cuda_device}")
        config["cuda_device"] = cuda_device

    wandb.config.update(config)

    # Initialize PyBullet in DIRECT mode (no GUI for headless)
    p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)

    # Configure base environment
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    PLANE_ID = p.loadURDF("plane.urdf")
    # TABLE_ID = p.loadURDF(
    #     "table/table.urdf",
    #     basePosition=[1.0, -0.2, 0.0],
    #     baseOrientation=p.getQuaternionFromEuler([0, 0, pi/8]),
    # )
    # CUBE_ID = p.loadURDF(
    #     "cube.urdf",
    #     basePosition=[0.85, -0.2, 1],
    #     globalScaling=0.05,
    # )
    ROBOT_ID = p.loadURDF(
        "kuka_iiwa/model.urdf",
        basePosition=[0.5, 0, 0],
        useFixedBase=True,
    )

    NUM_JOINTS = p.getNumJoints(ROBOT_ID)
    JOINT_INDEX = 6
    POSE = pi / 2
    p.resetJointState(ROBOT_ID, JOINT_INDEX, POSE)

    create_rgb_axes(p)

    # CUBE_BASE_POSITION = [1,-2,0.2]
    # CUBE_EULER_ORIENTATION = [0,0,0]

    # CUBE_ID = p.loadURDF(
    #     "cube.urdf",
    #     basePosition=CUBE_BASE_POSITION,
    #     baseOrientation=p.getQuaternionFromEuler(CUBE_EULER_ORIENTATION),
    #     globalScaling=0.1,
    # )
    # p.changeVisualShape(
    #     objectUniqueId=CUBE_ID,
    #     rgbaColor=[0, 0, 1, 1],
    #     linkIndex=-1
    # )

    print("PyBullet physics client initialized successfully!")

    # Set up camera parameters
    width, height = 640, 480
    fov, aspect, near, far = 60, width / height, 0.1, 100
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[2, 2, 2],
        cameraTargetPosition=[0, 0, 0],
        cameraUpVector=[0, 0, 1],
    )
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    # Capture a frame
    img_arr = p.getCameraImage(
        width,
        height,
        view_matrix,
        projection_matrix,
        renderer=p.ER_TINY_RENDERER,  # default: renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    rgb_array = img_arr[2][:, :, :3]  # Extract RGB channels (remove alpha)

    # Log frame to wandb
    wandb.log(
        {
            "camera_frame": wandb.Image(
                rgb_array, caption=f"KUKA Arm pose: {POSE} at joint {JOINT_INDEX}"
            )
        }
    )
    wandb.log({"joint_index": JOINT_INDEX, "pose": POSE})
    # wandb.log({
    #     "cube_base_position": p.getBasePositionAndOrientation(CUBE_ID)[0],
    #     "cube_base_orientation": p.getBasePositionAndOrientation(CUBE_ID)[1],
    #     "cube_base_orientation_euler": p.getEulerFromQuaternion(p.getBasePositionAndOrientation(CUBE_ID)[1]),
    # })
    print("Camera frame uploaded to wandb")

    p.disconnect()
    print("Test completed successfully!")

    # Finish wandb run
    wandb.finish()


@app.function(
    image=image,
    gpu="T4",  # Use NVIDIA T4 GPU
    timeout=3600,  # 1 hour timeout
    mounts=[src_mount],
)
def run_rl(wandb_api_key: str) -> None:
    import imageio
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from src.environments import KukaEnv, RewardLoggerCallback, _render_camera_view

    os.environ["WANDB_API_KEY"] = wandb_api_key
    wandb.login(key=wandb_api_key, relogin=True)
    wandb.init(
        entity="larryl729-team",
        project="pybullet-rl",
        name="reinforcement-learning-run",
    )

    class VideoRecorderCallback(BaseCallback):
        def __init__(self, record_freq: int = 5000, verbose: int = 0):
            super().__init__(verbose)
            self.record_freq = record_freq
            self.video_count = 0

        def _on_step(self) -> bool:
            if self.n_calls % self.record_freq == 0:
                self._record_video()
            return True

        def _record_video(self) -> None:
            frames = []
            obs, info = self.training_env.envs[0].env.reset()
            for i in range(1000):
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.training_env.envs[
                    0
                ].env.step(action)
                if i % 8 == 0:
                    frames.append(_render_camera_view())
                if terminated or truncated:
                    break

            video_path = f"robot_demo_{self.video_count}.mp4"
            imageio.mimsave(video_path, frames, fps=30)
            wandb.log(
                {
                    "video": wandb.Video(video_path),
                    "video_timestep": self.n_calls,
                }
            )
            self.video_count += 1

    env = KukaEnv()
    env.reset()

    reward_callback = RewardLoggerCallback()
    video_callback = VideoRecorderCallback(record_freq=5000)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=1000,
        batch_size=64,
        learning_rate=1e-4,
        n_epochs=10,
    )
    model.learn(total_timesteps=1e7, callback=[reward_callback, video_callback])

    # record a final video demo of the trained policy
    frames = []
    obs, info = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if i % 8 == 0:
            frames.append(_render_camera_view())
        if terminated or truncated:
            break

    # save frames to video
    imageio.mimsave("robot_demo_final.mp4", frames, fps=30)

    wandb.log({"video_final": wandb.Video("robot_demo_final.mp4")})

    env.close()


@app.local_entrypoint()
def main() -> None:
    """Local entrypoint to run the Modal function."""
    from dotenv import load_dotenv

    load_dotenv()

    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        raise ValueError("WANDB_API_KEY not found in environment")

    # Get mode from environment variable, default to "pybullet"
    mode = os.getenv("MODE", "pybullet")
    if mode not in ["pybullet", "rl"]:
        raise ValueError(f"Invalid MODE: {mode}. Must be 'pybullet' or 'rl'")

    if mode == "pybullet":
        run_pybullet.remote(wandb_api_key)
    elif mode == "rl":
        run_rl.remote(wandb_api_key)
