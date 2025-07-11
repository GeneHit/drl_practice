#!/usr/bin/env python3
"""CLI script to launch TensorBoard for DRL training logs.

This script reads a configuration file, extracts the log directory,
and launches TensorBoard to visualize training metrics.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from hands_on.utils.file_utils import load_config_from_json


def extract_log_dir_from_config(config_path: str) -> str:
    """Extract the TensorBoard log directory from configuration file.

    Args:
        config_path: Path to the configuration JSON file

    Returns:
        str: Path to the TensorBoard log directory

    Raises:
        KeyError: If output_params or output_dir is missing from config
        FileNotFoundError: If config file doesn't exist
    """
    try:
        cfg_data: dict[str, Any] = load_config_from_json(config_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except Exception as e:
        raise ValueError(f"Failed to load configuration: {e}")

    # Extract output_dir from config
    if "output_params" not in cfg_data:
        raise KeyError("Missing 'output_params' section in configuration")

    output_params = cfg_data["output_params"]
    if "output_dir" not in output_params:
        raise KeyError("Missing 'output_dir' in output_params section")

    output_dir = output_params["output_dir"]

    # Construct log directory path (training logs are in output_dir/runs)
    log_dir = os.path.join(output_dir, "runs")

    return log_dir


def launch_tensorboard(log_dir: str, port: int = 6006, host: str = "localhost") -> None:
    """Launch TensorBoard with the specified log directory.

    Args:
        log_dir: Path to the TensorBoard log directory
        port: Port number for TensorBoard server (default: 6006)
        host: Host address for TensorBoard server (default: localhost)

    Raises:
        FileNotFoundError: If log directory doesn't exist
        subprocess.CalledProcessError: If TensorBoard fails to start
    """
    # Check if log directory exists
    log_path = Path(log_dir)
    if not log_path.exists():
        raise FileNotFoundError(
            f"Log directory does not exist: {log_dir}\n"
            f"Make sure you have run training first to generate TensorBoard logs."
        )

    # Check if log directory has any log files
    log_files = list(log_path.rglob("*"))
    if not log_files:
        print(f"Warning: Log directory is empty: {log_dir}")
        print("Make sure training has been completed to generate logs.")

    # Construct TensorBoard command
    cmd = [
        "tensorboard",
        "--logdir",
        log_dir,
        "--port",
        str(port),
        "--host",
        host,
    ]

    print("Starting TensorBoard...")
    print(f"Log directory: {log_dir}")
    print(f"TensorBoard URL: http://{host}:{port}")
    print(f"Command: {' '.join(cmd)}")
    print("\nPress Ctrl+C to stop TensorBoard")

    try:
        # Launch TensorBoard (this will block until user stops it)
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nTensorBoard stopped by user")
    except FileNotFoundError:
        print("\nError: TensorBoard not found. Please install TensorBoard:")
        print("pip install tensorboard")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nError: TensorBoard failed to start: {e}")
        sys.exit(1)


def main() -> None:
    """Main function for the TensorBoard CLI."""
    parser = argparse.ArgumentParser(
        description="Launch TensorBoard for DRL training logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch TensorBoard for Q-learning training logs
  python hands_on/tensorboard_cli.py --config hands_on/exercise1_q_learning/config_taxi.json

  # Launch TensorBoard on custom port
  python hands_on/tensorboard_cli.py --config hands_on/exercise2_dqn/obs_1d_config.json --port 6007

  # Launch TensorBoard accessible from other machines
  python hands_on/tensorboard_cli.py --config config.json --host 0.0.0.0
        """,
    )

    parser.add_argument("--config", type=str, required=True, help="Path to configuration JSON file")

    parser.add_argument(
        "--port", type=int, default=6006, help="Port number for TensorBoard server (default: 6006)"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host address for TensorBoard server (default: localhost)",
    )

    args = parser.parse_args()

    try:
        # Extract log directory from config
        log_dir = extract_log_dir_from_config(args.config)

        # Launch TensorBoard
        launch_tensorboard(log_dir, port=args.port, host=args.host)

    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
