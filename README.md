# bomberman_rl
Setup for a project/competition amongst students to train a winning Reinforcement Learning agent for the classic game Bomberman.

## setup conda env

create conda env by using 

```
conda create --name bomberman python=3.8
pip install numpy tensorflow tqdm keras pygame tables matplotlib
```

## play and train

```python main.py play --my-agent dqn```

starts the game and executes our own trained agent

```python main.py play --my-agent dqn --train 1 --no-gui --n-rounds=200```

starts the training process (```python evaluate_training.py create``` must be called before training if evaluation is enabled)

## evaluate training

1. In parameters.py: choose FILENAME, EVALUATION=True
2. ```python evaluate_training.py create```
  Attention: Already existing files are overwritten!
3. train model as described above
4. ```python evaluate_training.py``` plots the data
