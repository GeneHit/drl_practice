# Command Guide


## 1. Create the python env
First, create the environment with the `python 3.10`. For conda, use
```
conda create -n drl python=3.10
```
where you can rename the environment name `drl`. Then enter the env:
```
conda activate drl
```

Second, install the required libs:
```
python -m pip install -r requirements.txt
```
Since Box2D (installed via gymnasium[box2d]) is a C++ extension, we do not
recommend using the `uv` environment tool that for pure pyhton env.


## 2. Exercise Command
All commands are run under the project root. If it can not fing the module path,
you can try (add project root to the python path):
```
export PYTHONPATH=$PYTHONPATH:$(pwd)
```
### Good Code
To ensure your code is high quality, please use the pre-commit:
- when you submit a `git commit`:
    - it will run a range of checks.
    - include self-add mypy and pytest check.
- commit will be submited when it passes all checks.

Run below command in terminal to install the pre-commit for `git commit`.
```
pre-commit install
```
where you can uninstall it by `pre-commit unstall`.

You also can run mypy and pytest seperately. Some commands:
```
mypy .
mypy hands_on/exercise1_q_learning/q_learing_train.py
pytest .
pytest -s common/
pytest -s $FOLDER$ -k $function_name$
```

### Sandbox
Check the specified environment:
1. Change the env to what you want in the [env_test.py](./sandbox/env_test.py) file
2. run the below command to see whether your laptop work.
    ```
    python hands_on/sandbox/env_test.py
    ```

Plot the training rewards:
1. change the file path in [plot_train_rewards.py](./sandbox/plot_train_rewards.py)
2. run
    ```
    python hands_on/sanbox/plot_train_rewards.py
    ```


### Exercise 1: Q Learning

The unified main script for `taxi` (can change to `frozen_lake`).

Train the q_table from zeros:
```
python hands_on/exercise1_q_learning/q_main.py --config hands_on/exercise1_q_learning/config_taxi.json
```
or
```
python hands_on/exercise1_q_learning/q_main.py --config hands_on/exercise1_q_learning/config_frozen_lake.json
```

Train without playing/video generation:
```
python hands_on/exercise1_q_learning/q_main.py --config hands_on/exercise1_q_learning/config_taxi.json --skip_play
```

Generate a replay video only (to `results/exercise1_q/` folder):
```
python hands_on/exercise1_q_learning/q_main.py --config hands_on/exercise1_q_learning/config_taxi.json --mode play_only
```

Push model to HuggingFace hub (with video generation):
```
python hands_on/exercise1_q_learning/q_main.py --config hands_on/exercise1_q_learning/config_taxi.json --mode push_to_hub --username $YOUR_USERNAME$
```

Push model to HuggingFace hub (without video generation):
```
python hands_on/exercise1_q_learning/q_main.py --config hands_on/exercise1_q_learning/config_taxi.json --mode push_to_hub --username $YOUR_USERNAME$ --skip_play
```

**Note:**:Replace `$YOUR_USERNAME$` with your HuggingFace account name.

For hub operations, you need to login first:
- run `huggingface-cli login` in terminal.
- go to the HuggingFace website, copy your access token and paste it in terminal.


### Exercise 2: DQN

The unified main script for `2d` (change to `1d` for the discre observation).

Train:
```
python hands_on/exercise2_dqn/dqn_main.py --config hands_on/exercise2_dqn/obs_2d_config.json
```
or
```
python hands_on/exercise2_dqn/dqn_main.py --config hands_on/exercise2_dqn/obs_1d_config.json
```

Train without playing/video generation:
```
python hands_on/exercise2_dqn/dqn_main.py --config hands_on/exercise2_dqn/obs_2d_config.json --skip_play
```

Generate a replay video only:
```
python hands_on/exercise2_dqn/dqn_main.py --config hands_on/exercise2_dqn/obs_2d_config.json --mode play_only
```

Push model to HuggingFace hub (with video generation):
```
python hands_on/exercise2_dqn/dqn_main.py --config hands_on/exercise2_dqn/obs_2d_config.json --mode push_to_hub --username $YOUR_USERNAME$
```

Push model to HuggingFace hub (without video generation):
```
python hands_on/exercise2_dqn/dqn_main.py --config hands_on/exercise2_dqn/obs_2d_config.json --mode push_to_hub --username $YOUR_USERNAME$ --skip_play
```

**Note:** Replace `$YOUR_USERNAME$` with your HuggingFace account name.
