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
python hands_on/exercise1_q_learning/q_learning_train.py
```
Or train it from your previous checkpoint:
```
python hands_on/exercise1_q_learning/q_learning_train.py --model_pathname results/exercise1_q_learning/q_table.pkl
```
Generate a replay video only (to `results/exercise1_q_learning/` folder):
```
python hands_on/exercise1_q_learning/q_learning_play.py
```
Only push the model and result to HuggingFace hub:
- login
    - run `huggingface-cli login` in terminal.
    - go to the HuggingFace website, copy your access token and paste it in terminal.
```
python hands_on/exercise1_q_learning/q_learning_play.py --push_to_hub --skip_render --username $YOUR_USERNAMER$
```
where you need to replace `$YOUR_USERNAMER$` with your HuggingFace account name.

Generate the video and push the model/result to the hub in the mean time:
```
python hands_on/exercise1_q_learning/q_learning_play.py --push_to_hub --username $YOUR_USERNAMER$
```
