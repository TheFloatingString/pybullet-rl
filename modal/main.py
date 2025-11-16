import modal
import os


app = modal.App("pybullet-rl")

# Create image with pybullet, torch, and stable-baselines3
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pybullet",
    "torch",
    "stable-baselines3",
    "wandb",
)


@app.function(
    image=image,
    gpu="T4",  # Use NVIDIA T4 GPU
    timeout=3600,  # 1 hour timeout
)
def run_pybullet(wandb_api_key: str) -> None:
    """Run PyBullet on GPU instance with torch and stable-baselines3."""
    import os

    import pybullet as p
    import pybullet_data
    import torch
    import wandb
    from math import pi

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
    # ROBOT_ID = p.loadURDF(
    #     "kuka_iiwa/model.urdf",
    #     basePosition=[1.4, 1, 0.6],
    #     useFixedBase=True,
    # )

    # Define the origin and the length of the axes
    origin = [0, 0, 0.5]  # Start drawing at z=0.5
    axis_length = 1.0
    arrow_radius = 0.02
    arrow_head_radius = 0.04
    arrow_head_length = 0.15

    # --- Create XYZ (RGB) Arrows using visual shapes ---

    # X-axis (Red) - arrow pointing in +X direction
    x_shaft = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=arrow_radius,
        length=axis_length - arrow_head_length,
        rgbaColor=[1, 0, 0, 1],
    )
    x_head = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=arrow_head_radius,
        length=arrow_head_length,
        rgbaColor=[1, 0, 0, 1],
    )
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=x_shaft,
        basePosition=[
            origin[0] + (axis_length - arrow_head_length) / 2,
            origin[1],
            origin[2],
        ],
        baseOrientation=p.getQuaternionFromEuler([0, pi / 2, 0]),
    )
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=x_head,
        basePosition=[
            origin[0] + axis_length - arrow_head_length / 2,
            origin[1],
            origin[2],
        ],
        baseOrientation=p.getQuaternionFromEuler([0, pi / 2, 0]),
    )

    # Y-axis (Green) - arrow pointing in +Y direction
    y_shaft = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=arrow_radius,
        length=axis_length - arrow_head_length,
        rgbaColor=[0, 1, 0, 1],
    )
    y_head = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=arrow_head_radius,
        length=arrow_head_length,
        rgbaColor=[0, 1, 0, 1],
    )
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=y_shaft,
        basePosition=[
            origin[0],
            origin[1] + (axis_length - arrow_head_length) / 2,
            origin[2],
        ],
        baseOrientation=p.getQuaternionFromEuler([pi / 2, 0, 0]),
    )
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=y_head,
        basePosition=[
            origin[0],
            origin[1] + axis_length - arrow_head_length / 2,
            origin[2],
        ],
        baseOrientation=p.getQuaternionFromEuler([pi / 2, 0, 0]),
    )

    # Z-axis (Blue) - arrow pointing in +Z direction
    z_shaft = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=arrow_radius,
        length=axis_length - arrow_head_length,
        rgbaColor=[0, 0, 1, 1],
    )
    z_head = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=arrow_head_radius,
        length=arrow_head_length,
        rgbaColor=[0, 0, 1, 1],
    )
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=z_shaft,
        basePosition=[
            origin[0],
            origin[1],
            origin[2] + (axis_length - arrow_head_length) / 2,
        ],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    )
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=z_head,
        basePosition=[
            origin[0],
            origin[1],
            origin[2] + axis_length - arrow_head_length / 2,
        ],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    )

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
    wandb.log({"camera_frame": wandb.Image(rgb_array, caption="PyBullet Camera View")})
    print("Camera frame uploaded to wandb")

    p.disconnect()
    print("Test completed successfully!")

    # Finish wandb run
    wandb.finish()


@app.local_entrypoint()
def main() -> None:
    """Local entrypoint to run the Modal function."""
    from dotenv import load_dotenv

    load_dotenv()

    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        raise ValueError("WANDB_API_KEY not found in environment")

    run_pybullet.remote(wandb_api_key)
