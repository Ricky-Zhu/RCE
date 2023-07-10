import numpy as np
from collections import deque
import random
import gym


class N_step_traj:
    """
    a container used to store n steps sub-trajs. can return n-steps states, actions, rewards and final state
    which will be given if done or reach the end of the n-steps traj
    """

    def __init__(self, n_steps=10):
        self.n_steps = n_steps
        self.reset()

    @property
    def length(self):
        return len(self.states)

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.final_state = None

    def add(self, state, action, reward, final_sate, done):
        if done:
            return True
        else:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)
            flag = self.complete(done)
            if flag:
                self.final_state = final_sate
            return flag

    def dump(self):
        self.states = np.asarray(self.states)
        self.actions = np.asarray(self.actions)
        self.rewards = np.asarray(self.rewards)
        self.dones = np.asarray(self.dones)
        self.final_state = np.asarray(self.final_state)

        return self.states, self.actions, self.rewards, self.dones, self.final_state

    def complete(self, done):
        flag = done or self.length == self.n_steps
        return flag


class UniformReplayBuffer:
    def __init__(self,
                 max_size=int(1e6),
                 ):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add_n_step_traj(self, traj):
        self.buffer.append(traj)

    def sample(self, batch_size):
        batch_n_steps_traj = random.sample(self.buffer, batch_size)
        return np.array(batch_n_steps_traj)


class ExpertReplayBuffer:
    def __init__(self,env,example_num=200, terminal_offset=50):
        dataset = env.get_dataset()
        terminals = np.where(dataset['terminals'])[0]
        expert_obs = np.concatenate(
            [dataset['observations'][t - terminal_offset:t] for t in terminals],
            axis=0)
        indices = np.random.choice(
            len(expert_obs), size=example_num, replace=False)
        self.expert_obs = expert_obs[indices]
        self.index = np.arange(example_num)

    def sample(self, batch_size):
        temp_ind = np.random.choice(self.index, batch_size, replace=False)
        batch_obs = self.expert_obs[temp_ind]
        return batch_obs

    @property
    def buffer_size(self):
        return self.expert_obs[0]








