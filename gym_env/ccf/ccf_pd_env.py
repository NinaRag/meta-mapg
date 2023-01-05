import os

import random
import pickle

import gym
import numpy as np
from gym.spaces import Discrete, Tuple

from misc.utils import to_onehot


class CCFPdEnv(gym.Env):
    """
    Base class for two agent prisoner's dilemma game with CCF actions
    Each agent can either cooperate or defect or cooperate conditionally

    Args:
        args (argparse): Python argparse that contains arguments
    """

    def __init__(self, args):
        super(CCFPdEnv, self).__init__()
        assert args.n_agent == 2, "Only two agents are supported in this domain"

        self.args = args
        self.payoff_matrix = np.array([
            [[1, 1], [1, 1], [0, 3]],
            [[1, 1], [2, 2], [2, 2]],
            [[3, 0], [2, 2], [2, 2]]
        ], dtype=np.float32)

        # S0 + 3*3 outcomes
        self.observation_space = [Discrete(10) for _ in range(2)]
        self.states = np.arange(start=1, stop=10, step=1, dtype=np.int32)
        self.action_space = Tuple([Discrete(3) for _ in range(2)])

    def reset(self):
        obs = np.zeros(self.args.traj_batch_size, dtype=np.int32)
        obs = to_onehot(obs, dim=10)
        return obs

    def step(self, actions):
        state = self._action_to_state(actions)
        assert len(state.shape) == 1, "Shape should be (traj_batch_size,)"

        # Get observation
        obs = self.states[state]
        obs = to_onehot(obs, dim=10)

        # Get reward
        rewards = []
        for i_agent in range(2):
            agent_i_matrix = self.payoff_matrix[:, :, i_agent].reshape(-1)
            rewards.append(agent_i_matrix[state])

        # Get done
        done = False

        return obs, rewards, done, {}

    def render(self, mode='human', close=False):
        raise NotImplementedError()

    def _action_to_state(self, actions):
        assert actions[0].shape == actions[1].shape
        action0, action1 = actions
        # state is the (3,3) payoff matrix flattened to 9,
        # Therefore the index is row *3 + col
        state = 3 * action0 + action1
        return state

    @staticmethod
    def sample_personas(is_train, is_val=True, path="./"):
        path = path + "pretrain_model/CCF-PD-v0/"

        if is_train:
            with open(path + "d/train", "rb") as fp:
                defective_personas = pickle.load(fp)
            with open(path + "cc/train", "rb") as fp:
                conditional_personas = pickle.load(fp)
            with open(path + "c/train", "rb") as fp:
                cooperative_personas = pickle.load(fp)
            return random.choices(defective_personas + conditional_personas + cooperative_personas, k=1)
        else:
            if is_val:
                with open(path + "d/val", "rb") as fp:
                    defective_personas = pickle.load(fp)
                with open(path + "cc/val", "rb") as fp:
                    conditional_personas = pickle.load(fp)
                with open(path + "c/val", "rb") as fp:
                    cooperative_personas = pickle.load(fp)
            else:
                with open(path + "d/test", "rb") as fp:
                    defective_personas = pickle.load(fp)
                with open(path + "cc/test", "rb") as fp:
                    conditional_personas = pickle.load(fp)
                with open(path + "c/test", "rb") as fp:
                    cooperative_personas = pickle.load(fp)
            return defective_personas + conditional_personas + cooperative_personas

    @staticmethod
    def generate_personas(n_train=200, n_val=20, n_test=20):
        path = "./pretrain_model/CCF-PD-v0/"

        # Cooperate, conditionally cooperate, defect
        for persona_type in ["c", "cc", "d"]:
            personas = []
            for _ in range(n_train + n_val + n_test):
                if persona_type == "c":
                    c_prob = np.random.uniform(low=1. / 3., high=1., size=(10,))
                    cc_prob = (1. - c_prob) / 2.
                    d_prob = (1. - c_prob) / 2.
                    assert np.array_equal(
                        np.argmax(np.stack([c_prob, cc_prob, d_prob], axis=1), axis=1), np.zeros((10,)))
                elif persona_type == "cc":
                    cc_prob = np.random.uniform(low=1. / 3., high=1., size=(10,))
                    d_prob = (1. - cc_prob) / 2.
                    c_prob = (1. - cc_prob) / 2.
                    assert np.array_equal(
                        np.argmax(np.stack([cc_prob, d_prob, c_prob], axis=1), axis=1), np.zeros((10,)))
                elif persona_type == "d":
                    d_prob = np.random.uniform(low=1. / 3., high=1., size=(10,))
                    c_prob = (1. - d_prob) / 2.
                    cc_prob = (1. - d_prob) / 2.
                    assert np.array_equal(
                        np.argmax(np.stack([d_prob, c_prob, cc_prob], axis=1), axis=1), np.zeros((10,)))
                else:
                    raise ValueError()

                assert np.sum(c_prob + cc_prob + d_prob) == 10.
                assert np.allclose((c_prob + cc_prob + d_prob), np.ones((10,)))

                persona = np.log(np.stack([c_prob, cc_prob, d_prob], axis=1))
                personas.append(persona)

            with open(path + persona_type + "/train", "wb") as fp:
                pickle.dump(personas[:n_train], fp)

            with open(path + persona_type + "/val", "wb") as fp:
                pickle.dump(personas[n_train:n_train + n_val], fp)

            with open(path + persona_type + "/test", "wb") as fp:
                pickle.dump(personas[n_train + n_val:], fp)
