from pathlib import Path
from typing import Any

from hands_on.exercise2_dqn.dqn_exercise import EnvType
from hands_on.utils.hub_play_utils import (
    get_env_name_and_metadata,
    push_to_hub,
)
from practice.base.config import ArtifactConfig, BaseConfig


def push_to_hub_generic(config: BaseConfig, env: EnvType, username: str) -> None:
    """Generic function to push any model to Hugging Face Hub.

    Args:
        config: Configuration data containing env_params, hub_params, output_params
        env: Environment
        username: Hugging Face username
    """
    env_id = config.env_config.env_id
    algorithm_name = config.artifact_config.algorithm_name
    try:
        # Get metadata with algorithm-specific tags
        metadata = get_env_name_and_metadata(
            env_id=env_id,
            env=env,
            algorithm_name=algorithm_name.lower(),
            extra_tags=config.artifact_config.extra_tags or [],
        )

        # Create repo_id
        repo_id = f"{username}/{config.artifact_config.repo_id}"

        # Create model card with algorithm-specific content
        model_card = f"""
    # **{algorithm_name}** Agent playing **{env_id}**
    This is a trained model of a **{algorithm_name}** agent playing **{env_id}**.

    ## Usage

    model = load_from_hub(repo_id="{repo_id}", filename="{config.artifact_config.model_filename}")

    {config.artifact_config.usage_instructions}
    env = gym.make("{env_id}")
    ...
    """

        # Push to hub
        push_model_to_hub(
            repo_id=repo_id,
            artifact_config=config.artifact_config,
            model_card=model_card,
            metadata=metadata,
        )
    finally:
        env.close()


def push_model_to_hub(
    repo_id: str,
    artifact_config: ArtifactConfig,
    model_card: str,
    metadata: dict[str, Any],
) -> None:
    """Common function to push model files to hub."""
    output_dir = Path(artifact_config.output_dir)

    push_to_hub(
        repo_id=repo_id,
        model_card=model_card,
        file_pathnames=[
            str(output_dir / artifact_config.model_filename),
            str(output_dir / artifact_config.params_filename),
            str(output_dir / artifact_config.replay_video_filename),
        ],
        eval_result_pathname=str(output_dir / artifact_config.eval_result_filename),
        metadata=metadata,
        local_repo_path=str(output_dir),
        copy_file=False,
    )
