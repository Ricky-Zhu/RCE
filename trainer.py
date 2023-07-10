import gym
import d4rl
from agent import RceAgent
from argments import get_args
from replaybuffer import ExpertReplayBuffer
import torch
import numpy as np
import random

if __name__ == "__main__":
    def set_seeds(args, rank=0):
        # set seeds for the numpy
        np.random.seed(args.seed + rank)
        # set seeds for the random.random
        random.seed(args.seed + rank)
        # set seeds for the pytorch
        torch.manual_seed(args.seed + rank)
        if args.device == 'cuda':
            torch.cuda.manual_seed(args.seed + rank)
            torch.cuda.manual_seed_all(args.seed + rank)


    args = get_args()
    set_seeds(args)
    env = gym.make(args.env_name)
    eval_env = gym.make(args.env_name)
    env.seed(args.seed)
    eval_env.seed(args.seed + 1)
    expert_buffer = ExpertReplayBuffer(env=env, example_num=500)

    env_params = {'max_action': env.action_space.high[0],
                  'state_dim': env.observation_space.shape[0],
                  'action_dim': env.action_space.shape[0]}

    agent = RceAgent(env, eval_env, args, env_params, expert_examples_buffer=expert_buffer)
    agent.learn()
