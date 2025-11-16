import modal

app = modal.App("pybullet-rl")

# Create image with pybullet, torch, and stable-baselines3
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pybullet",
    "torch",
    "stable-baselines3",
)


@app.function(
    image=image,
    gpu="T4",  # Use NVIDIA T4 GPU
    timeout=3600,  # 1 hour timeout
)
def run_pybullet() -> None:
    """Run PyBullet on GPU instance with torch and stable-baselines3."""
    import pybullet as p
    import torch

    print("PyBullet GPU Instance Started!")
    print(f"PyBullet version: {p.getAPIVersion()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Initialize PyBullet in DIRECT mode (no GUI for headless)
    p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)

    print("PyBullet physics client initialized successfully!")

    p.disconnect()
    print("Test completed successfully!")


@app.local_entrypoint()
def main() -> None:
    """Local entrypoint to run the Modal function."""
    run_pybullet.remote()
