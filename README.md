# fantasy-bball-draft-rl
RL agent for 9cat fantasy bball draft

Github repo: https://github.com/arthurhhe/fantasy-bball-draft-rl/tree/main

## Training commands
Discrete SAC
```
python -m ac.main
```
Optional flag `--roster_size` e.g. `--roster_size 10`

DQN
```
python -m dqn.main
```

PPO
```
python -m ppo.main
```