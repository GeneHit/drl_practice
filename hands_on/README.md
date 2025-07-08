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

### Test a Specified env
Change the env to what you want in the [env_test.py](./env_test.py) file,
and run the below command to see whether your laptop work.
```
python hands_on/env_test.py
```
It works if no error occur.


### Exercise 1: Q Learning
Train the q_table from zeros:
```
python hands_on/exercise1_q_learning/q_learning_train.py --config hands_on/exercise1_q_learning/config_taxi.json
```
or
```
python hands_on/exercise1_q_learning/q_learning_train.py --config hands_on/exercise1_q_learning/config_frozen_lake.json
```
You can change the configuration in the `json` file.

Generate a replay video only (to `results/exercise1_q/` folder):
```
python hands_on/exercise1_q_learning/q_learning_play.py --config hands_on/exercise1_q_learning/config_taxi.json
```
Only push the model and result to HuggingFace hub:
- login
    - run `huggingface-cli login` in terminal.
    - go to the HuggingFace website, copy your access token and paste it in terminal.
```
python hands_on/exercise1_q_learning/q_learning_play.py --config hands_on/exercise1_q_learning/config_taxi.json --push_to_hub --skip_play --username $YOUR_USERNAMER$
```
where you need to replace `$YOUR_USERNAMER$` with your HuggingFace account name.

Generate the video and push the model/result to the hub in the mean time:
```
python hands_on/exercise1_q_learning/q_learning_play.py --config hands_on/exercise1_q_learning/config_taxi.json --push_to_hub --username $YOUR_USERNAMER$
```

### Exercise 2: DQN
Train:
```
python hands_on/exercise2_dqn/dqn_train.py --config hands_on/exercise2_dqn/obs_1d_config.json
```
or
```
python hands_on/exercise2_dqn/dqn_train.py --config hands_on/exercise2_dqn/obs_2d_config.json
```
Train with multiprocessing multi envs:
```
python hands_on/exercise2_dqn/dqn_envs_train.py --config hands_on/exercise2_dqn/obs_1d_envs_config.json

python hands_on/exercise2_dqn/dqn_envs_train.py --config hands_on/exercise2_dqn/obs_2d_envs_config.json
```
