"""
Main entry point for our Fake News Detection project.

Here we simply:
1. Load the YAML config
2. Set up all folder paths
3. Call the training function
"""

import yaml
import torch

from src.utils.paths import ProjectPaths
from src.training.train import train_model


def load_config(path: str = "config.yaml"):
    """Load configuration from YAML file."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    # Load config
    config = load_config()

    # Prepare all paths using our small helper class
    paths = ProjectPaths(
        data_raw=config["paths"]["data_raw"],
        data_processed=config["paths"]["data_processed"],
        figures_dir=config["paths"]["figures_dir"],
        results_dir=config["paths"]["results_dir"],
        outputs_dir=config["paths"]["outputs_dir"],
        checkpoints_dir=config["paths"]["checkpoints_dir"],
        predictions_dir=config["paths"]["predictions_dir"],
    )
    paths.make_dirs()

    # Decide device (GPU if available)
    device = config["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA not available on this machine. Falling back to CPU.")
        device = "cpu"
    print(f"Using device: {device}")

    # Start training
    test_metrics = train_model(config=config, paths=paths, device=device)

    print("\n==============================")
    print("✅ Training + evaluation done")
    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")
    print("==============================")