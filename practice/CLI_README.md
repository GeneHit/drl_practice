# Practice RL Exercises - Unified CLI

This document describes the unified Command Line Interface (CLI) for running practice reinforcement learning exercises.

## Overview

The unified CLI (`practice/cli.py`) provides a single entry point for running different RL exercises with Python configuration files. It supports multiple modes of operation and can dynamically load exercise configurations.

## Usage

```bash
python practice/cli.py --config <config_file> [--mode <mode>] [--username <username>] [--skip_play]
```

### Arguments

- `--config`: **Required.** Path to Python configuration file (e.g., `practice/exercise4_curiosity/config_mountain_car.py`)
- `--mode`: Mode of operation. Choices: `train` (default), `play`, `push_to_hub`
- `--username`: Hugging Face username (required for `push_to_hub` mode)
- `--skip_play`: Skip playing the game and video generation (only for `push_to_hub` mode)

### Modes

#### Train Mode (Default)
Trains the model and includes evaluation with video generation.

```bash
# Train with evaluation and video generation
python practice/cli.py --config practice/exercise4_curiosity/config_cartpole.py
```

#### Play Mode
Only plays the game and generates video (no training). **Currently not implemented.**

```bash
python practice/cli.py --config practice/exercise4_curiosity/config_cartpole.py --mode play
```

#### Push to Hub Mode
Pushes the model to Hugging Face Hub with optional play/video generation. **Currently not implemented.**

```bash
# Push to hub with play/video generation
python practice/cli.py --config practice/exercise4_curiosity/config_cartpole.py --mode push_to_hub --username myuser

# Push to hub without play/video generation
python practice/cli.py --config practice/exercise4_curiosity/config_cartpole.py --mode push_to_hub --username myuser --skip_play
```

## Configuration Files

The CLI works with Python configuration files that must define two functions:

1. `get_app_config()` - Returns the application configuration object
2. `generate_context(config)` - Returns the context object with environment, network, optimizer, and trainer

### Example Configuration Structure

```python
def get_app_config() -> SomeConfig:
    """Get the application config."""
    return SomeConfig(
        device=device,
        # ... other config parameters
    )

def generate_context(config: SomeConfig) -> SomeContext:
    """Generate the context for the application."""
    # Create environment, network, optimizer, etc.
    return SomeContext(
        env=env,
        eval_env=eval_env,
        network=network,
        optimizer=optimizer,
        trainer_name=SomeTrainer,
        # ... other context parameters
    )
```

## Available Exercises

### Exercise 4: Curiosity-Driven Learning

- **Mountain Car**: `practice/exercise4_curiosity/config_mountain_car.py`
  - Uses RND (Random Network Distillation) and X-Direction Shaping rewards
  - Environment: MountainCar-v0

- **CartPole**: `practice/exercise4_curiosity/config_cartpole.py`
  - Uses RND (Random Network Distillation) reward
  - Environment: CartPole-v1

## Examples

```bash
# Train Mountain Car with curiosity-driven learning
python practice/cli.py --config practice/exercise4_curiosity/config_mountain_car.py

# Train CartPole without video generation
python practice/cli.py --config practice/exercise4_curiosity/config_cartpole.py --skip_play

# Show help
python practice/cli.py --help
```

## Error Handling

The CLI provides clear error messages for common issues:

- Missing configuration file
- Missing required functions in configuration
- Invalid mode/argument combinations
- Missing username for hub operations

## Implementation Notes

- The CLI automatically adds the project root to Python path for imports
- It uses dynamic module loading to import configuration files
- Type checking is enforced with mypy
- Circular import issues are resolved using `TYPE_CHECKING`

## Future Enhancements

- Complete implementation of `play` and `push_to_hub` modes
- Support for additional exercise types
- Integration with hands_on utilities for full functionality
- Batch processing of multiple configurations
