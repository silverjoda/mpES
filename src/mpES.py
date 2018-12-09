import numpy as np
import gym
import cma
import time
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import pytorch_ES
import multiprocessing as mp
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

T.set_num_threads(1)

class Baseline(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Baseline, self).__init__()
        self.fc1 = nn.Linear(obs_dim, act_dim)

    def forward(self, x):
        x = self.fc1(x)
        return x


def f_wrapper(env, policy, animate):
    def f(w):
        reward = 0
        done = False
        obs = env.reset()

        pytorch_ES.vector_to_parameters(torch.from_numpy(w), policy.parameters())

        while not done:

            # Get action from policy
            with torch.no_grad():
                act = policy(torch.from_numpy(np.expand_dims(obs, 0)))[0].numpy()

            # Step environment
            obs, rew, done, _ = env.step(act)

            if animate:
                env.render()

            reward += rew

        return -reward
    return f


def f_mp(args):
    env_name, policy, w = args
    env = gym.make(env_name)
    reward = 0
    done = False
    obs = env.reset()

    pytorch_ES.vector_to_parameters(torch.from_numpy(w), policy.parameters())

    while not done:

        # Get action from policy
        with torch.no_grad():
            act = policy(torch.from_numpy(np.expand_dims(obs, 0)))[0].numpy()

        # Step environment
        obs, rew, done, _ = env.step(act)

        reward += rew

    return -reward


def train_st(params):

    env_name, iters, n_hidden, animate = params

    # Make environment
    env = gym.make(env_name)

    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    policy = Baseline(obs_dim, act_dim)
    w = pytorch_ES.parameters_to_vector(policy.parameters()).detach().numpy()
    es = cma.CMAEvolutionStrategy(w, 0.5)


    f = f_wrapper(env, policy, animate)

    print("Env: {} Action space: {}, observation space: {}, N_params: {}, comments: ...".format(env_name, env.action_space.shape,
                                                                                  env.observation_space.shape, len(w)))

    ctr = 0
    try:
        while not es.stop():
            ctr += 1
            if ctr > iters:
                break
            X = es.ask()

            evals = [f(x) for x in X]
            es.tell(X, evals)
            es.disp()
    except KeyboardInterrupt:
        print("User interrupted process.")

    return es.result.fbest


def train_mt(params):

    env_name, iters, n_hidden, animate = params

    # Make environment
    env = gym.make(env_name)

    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    policy = Baseline(obs_dim, act_dim)
    w = pytorch_ES.parameters_to_vector(policy.parameters()).detach().numpy()
    es = cma.CMAEvolutionStrategy(w, 0.5)

    print("Env: {} Action space: {}, observation space: {}, N_params: {}, comments: ...".format(env_name, env.action_space.shape,
                                                                                  env.observation_space.shape, len(w)))

    ctr = 0
    try:
        while not es.stop():
            ctr += 1
            if ctr > iters:
                break
            X = es.ask()

            N = len(X)
            p = Pool(4)

            evals = p.map(f_mp, list(zip([env_name] * N, [policy] * N,  X)))

            #evals = [f(x) for x in X]
            es.tell(X, evals)
            es.disp()
    except KeyboardInterrupt:
        print("User interrupted process.")

    return es.result.fbest



env_name = "Hopper-v2"
t1 = time.clock()
train_st((env_name, 100, 7, False))
t2 = time.clock()
print("Elapsed time: {}".format(t2 - t1))
exit()
