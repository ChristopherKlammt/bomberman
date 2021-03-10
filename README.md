# bomberman_rl
Setup for a project/competition amongst students to train a winning Reinforcement Learning agent for the classic game Bomberman.

## setup conda env

create conda env by using 

```
conda create --name bomberman python=3.8
pip install numpy tensorflow tqdm keras pygame
```

## play and train

```python main.py play --my-agent robin-fleige_clara-hartmann_christopher-klammt```

starts the game and executes our own trained agent

```python main.py play --my-agent robin-fleige_clara-hartmann_christopher-klammt --train 1 --no-gui --n-rounds=200```

starts the training process
