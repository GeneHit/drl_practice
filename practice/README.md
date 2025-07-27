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
### Good Code Style
To ensure your code is high-quality, please use the pre-commit:
- when you submit a `git commit`:
    - it will run a range of checks.
    - include self-add mypy check.
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
    python practice/sandbox/env_test.py
    ```

#### TensorBoard
Launch TensorBoard to view training metrics with run the below command in terminal:
```
tensorboard --logdir=Exercise_Result_Folder
```
For example
```
tensorboard --logdir=results/exercise3_reinforce/cartpole/tensorboard
```

### Exercise Command
1. [q_learning](./exercise1_q/README.md)
2. [dqn](./exercise2_dqn/README.md)
3. [vanilla reinforce](./exercise3_reinforce/README.md)
4. [curiosity with enhanced reinforce](./exercise4_curiosity/README.md)
5. [A2C with GAE](./exercise5_a2c/README.md)
6. [A3C](./exercise6_a3c/README.md)
7. [PPO](./exercise7_ppo/README.md)
8. [TD3](./exercise8_td3/README.md)
9. [SAC](./exercise9_sac/README.md)
10. [PPO+Curiosity+DDP](./exercise10_ddp_ppo/README.md)
11. [SAC+PER+DDP](./exercise11_ddp_sac/README.md)


## Other

[CLI_README](./infos/CLI_README.md): the design document for the command cli.
