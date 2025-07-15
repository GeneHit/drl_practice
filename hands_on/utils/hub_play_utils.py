"""Common utilities for pushing models to Hugging Face Hub from play scripts."""

import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.repocard import metadata_eval_result, metadata_save


def push_to_hub(
    repo_id: str,
    model_card: str,
    file_pathnames: Sequence[str],
    eval_result_pathname: str,
    metadata: dict[str, Any],
    local_repo_path: str = "results/hub",
    copy_file: bool = True,
) -> None:
    """Push a model to the Hub.

    Parameters
    ----------
    repo_id: str
        The repository ID to push the model to. 'username/repo_name'
    model_card: str
        The model card to push to the Hub.
    file_pathnames: Sequence[str]
        The file pathnames to push to the Hub.
    eval_result_pathname: str
        The pathname of the evaluation result to push to the Hub. It will be
        copied to the local repository path.
    metadata: dict[str, Any]
        The metadata to push to the Hub.
    local_repo_path: str
        The local path to save the repository to.
    """
    assert "env_name" in metadata
    _, repo_name = repo_id.split("/")
    api = HfApi()

    # Step 1: Create the repo
    repo_url = api.create_repo(repo_id=repo_id, exist_ok=True)

    # Step 2: Download files
    repo_local_path = Path(snapshot_download(repo_id=repo_id, local_dir=local_repo_path))

    # Step 3: Create the model card
    readme_path = repo_local_path / "README.md"
    print(readme_path.exists())
    if not readme_path.exists():
        with readme_path.open("w", encoding="utf-8") as f:
            f.write(model_card)

    # Step 4: Add metrics and metadata to the model card
    # open eval_result_pathname
    with open(eval_result_pathname, "r") as f:
        eval_result = json.load(f)
    mean_reward = eval_result["mean_reward"]
    std_reward = eval_result["std_reward"]
    # Add metrics
    eval = metadata_eval_result(
        model_pretty_name=repo_name,
        task_pretty_name="reinforcement-learning",
        task_id="reinforcement-learning",
        metrics_pretty_name="mean_reward",
        metrics_id="mean_reward",
        metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
        dataset_pretty_name=metadata["env_name"],
        dataset_id=metadata["env_name"],
    )
    # Merges both dictionaries
    metadata = {**metadata, **eval}

    # Save our metrics to Readme metadata
    metadata_save(readme_path, metadata)

    # Step 5: copy video, model, parameters, eval_result. keep the original file name
    if copy_file:
        for file_pathname in file_pathnames:
            shutil.copy(file_pathname, repo_local_path / file_pathname.split("/")[-1])

        shutil.copy(eval_result_pathname, repo_local_path / eval_result_pathname.split("/")[-1])

    # Step 7. Push everything to the Hub
    api.upload_folder(
        repo_id=repo_id,
        folder_path=repo_local_path,
        path_in_repo=".",
        commit_message="test hub",
    )

    print("Pushed to the Hub. You can view your model here: ", repo_url)


def get_env_name_and_metadata(
    env_id: str,
    env: gym.Env[Any, Any],
    algorithm_name: str,
    extra_tags: Sequence[str] = (),
) -> dict[str, Any]:
    """Common function to generate environment name and metadata for hub uploads."""
    env_name = env_id
    if env.spec is not None:
        map_name = env.spec.kwargs.get("map_name")
        if map_name:
            env_name = env_name + "-" + map_name
        is_slippery = env.spec.kwargs.get("is_slippery", True)
        if not is_slippery:
            env_name += "-noSlippery"

    tags = [env_name, algorithm_name, "reinforcement-learning", "custom-implementation"]
    if extra_tags:
        tags.extend(extra_tags)

    metadata = {"env_name": env_name, "tags": tags}

    return metadata


def push_model_to_hub(
    repo_id: str,
    output_params: dict[str, Any],
    model_card: str,
    metadata: dict[str, Any],
) -> None:
    """Common function to push model files to hub."""
    output_dir = Path(output_params["output_dir"])

    push_to_hub(
        repo_id=repo_id,
        model_card=model_card,
        file_pathnames=[
            str(output_dir / output_params["model_filename"]),
            str(output_dir / output_params["params_filename"]),
            str(output_dir / output_params["train_result_filename"]),
            str(output_dir / output_params.get("replay_video_filename", "replay.mp4")),
        ],
        eval_result_pathname=str(output_dir / output_params["eval_result_filename"]),
        metadata=metadata,
        local_repo_path=str(output_dir / "hub"),
    )
