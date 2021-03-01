# bomberman_rl
Setup for a project/competition amongst students to train a winning Reinforcement Learning agent for the classic game Bomberman.

## setup conda env

export required libraries

```conda list -e > requirements.txt```

create conda env by using 

```conda create --name bomberman python=3.8 --file requirements.txt```

## play and train

```python main.py play --my-agent robin-fleige_clara-hartmann_christopher-klammt```

starts the game and executes our own trained agent

```python main.py play --my-agent robin-fleige_clara-hartmann_christopher-klammt --train 1````

starts the training process
