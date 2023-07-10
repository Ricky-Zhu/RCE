# Replacing Rewards with Examples: Example-Based Policy Search via Recursive Classiﬁcation

![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)  
This is the pytorch implementation of ["Replacing Rewards with Examples: Example-Based Policy Search via Recursive Classiﬁcation"](https://arxiv.org/abs/2103.12656).
The original tensorflow version could be found [here](https://github.com/google-research/google-research/tree/master/rce).

Currently only supports the training of env `door-human-v0`. The support of the training of other environments will come out subsequently.

## Requirements
- python 3.7
- register wandb account
- mujoco
- other packages can be found in `requirements.txt`

## Setup the environment
```shell
pip install -e .
git clone https://github.com/rail-berkeley/d4rl.git
cd d4rl
pip install -e .
```

## TODO List
- [ ] support the training of other envs in the metaworld.

## Reproduce experiments
All the arguments can be found in `argments.py`.
```shell
python trainer.py
```

