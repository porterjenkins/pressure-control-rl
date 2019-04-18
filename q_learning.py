import numpy as np
import math
import pickle
from vfa import LinearVFA, LstmVFA, MlpVFA
import torch
import memory_replay
import copy


class QLearner(object):
    def __init__(self, gamma, alpha):
        self.gamma = gamma # discount factor
        self.alpha = alpha # learning rate

        self.actions = [.25, .2, .15, .1, .05, 0, -.25, -.2, -.15, -.1, -.05]
        self.idx_action_map = dict(zip(range(len(self.actions)), self.actions))
        self.action_idx_map = dict(zip(self.actions, range(len(self.actions))))


    def feature_extractor(self, observation):
        pass

    def get_q_hat(self, s):
        pass
    def get_target(self, s):
        pass

    def get_random_action(self):

        return np.random.choice(self.actions, 1)


    def get_epsilon_greedy_action(self, s, eps=.2):

        alpha = np.random.uniform(0,1)
        if alpha < eps:
            action = self.get_random_action()[0]
        else:
            action = self.get_greedy_action(s)

        return action


    def dump_model(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self.q_func, f)

    def load_model(self, fname):
        with open(fname, 'rb') as f:
            self.q_func = pickle.load(f)

    def get_q_max(self, s):
        """
        Action that maximized expected reward of target function
        :param s:
        :return:
        """
        q_hat = self.get_target(s)
        if isinstance(q_hat, np.ndarray):
            return np.max(q_hat)
        else:
            return q_hat.max()

    def update_q_value(self, s, a, r, q_max):
        pass


class TabularQLearning(QLearner):

    def __init__(self, n_states, min_state_val, max_state_val, gamma=.5, alpha=.1):
        super().__init__(gamma, alpha)
        self.min_state_val = min_state_val
        self.max_state_val = max_state_val
        self.n_states = n_states
        self.state_buckets = np.linspace(min_state_val, max_state_val, n_states, endpoint=False)
        self.bin_width = self.state_buckets[1] - self.state_buckets[0]


        self.q_func = np.random.uniform(-10, 0, (n_states, len(self.actions)))

        self.gamma = gamma
        self.alpha = alpha

    def feature_extractor(self, observation):
        s = observation[-1]
        state_idx = (s - self.min_state_val) // self.bin_width
        return int(state_idx)


    def get_q_hat(self, s):
        return self.q_func[s, :]

    def get_target(self, s):
        return self.q_func[s, :]



    def update_q_value(self, s, a, r, q_max):
        state_idx = s
        action_idx = self.action_idx_map[a]

        curr_q = self.q_func[state_idx, action_idx]
        self.q_func[state_idx, action_idx] = curr_q + self.alpha*(r + self.gamma*q_max - curr_q)
        new_q = self.q_func[state_idx, action_idx]


        print("Q-value update ({:.4f}, {:.2f}): {:.4f} --> {:.4f}".format(s, a, curr_q, new_q))

    def get_greedy_action(self, s):
        q_hat = self.get_q_hat(s)
        action_idx = np.argmax(q_hat)
        return self.idx_action_map[action_idx]




class LinearQLearning(QLearner):
    def __init__(self,  n_features, gamma=.5, alpha=.1):
        super().__init__(gamma, alpha)
        self.n_features = n_features
        self.q_func = LinearVFA(n_features, n_actions=len(self.actions))
        self.optim = self.q_func.get_optimizer(lr=.1)
        self.target_func = copy.deepcopy(self.q_func)

    def update_target_net(self):
        self.target_func.load_state_dict(self.q_func.state_dict())

    def feature_extractor(self, observations, norm=True):
        """Take the last k observations of the prms series"""
        n = len(observations)
        if norm:
            # standardize (mean: 0, std: 1)
            mu = np.mean(observations)
            sig = np.std(observations)

            observations = (observations - mu) / sig

        if n >= self.n_features:
            x = np.array(observations[-self.n_features:])
        else:
            x = np.zeros(self.n_features)
            x[(self.n_features-n):self.n_features] = observations

        return x


    def get_q_hat(self, s):
        s = torch.from_numpy(s).type(torch.FloatTensor)
        q_hat = self.q_func.forward(s)
        return q_hat

    def get_target(self, s):
        s = torch.from_numpy(s).type(torch.FloatTensor)
        q_hat = self.target_func.forward(s)
        return q_hat


    def update_q_value(self, s, a, r, q_max):
        if len(memory_replay.memory) < memory_replay.BATCH_SIZE:
            return

        transitions = memory_replay.memory.sample(memory_replay.BATCH_SIZE)
        batch = memory_replay.Transition(*zip(*transitions))

        self.optim.zero_grad()

        action_idx = self.action_idx_map[a]
        q_val_curr = self.get_q_hat(s)[action_idx]
        expected_q_val = q_max*self.gamma + r

        loss = self.q_func.get_loss(q_val_curr=q_val_curr, q_val_expected=expected_q_val)

        loss.backward()
        self.optim.step()

    def get_greedy_action(self, s):
        q_hat = self.get_q_hat(s)
        action_idx = q_hat.max(0)[1].item()
        return self.idx_action_map[action_idx]

class MlpQlearning(LinearQLearning):
    def __init__(self, n_features, gamma=.5, alpha=.1):
        super().__init__(n_features, gamma, alpha)
        self.q_func = MlpVFA(n_features,  n_actions=len(self.actions))
        self.optim = self.q_func.get_optimizer(lr=.1)
        self.target_func = copy.deepcopy(self.q_func)


class LstmQLearning(LinearQLearning):
    def __init__(self, seq_size, hidden_dim, gamma=.5, alpha=.1):
        super().__init__(seq_size, gamma, alpha)
        self.hidden_dim = hidden_dim
        self.q_func = LstmVFA(seq_size, hidden_dim=hidden_dim, n_actions=len(self.actions))
        self.optim = self.q_func.get_optimizer(lr=.1)
        self.target_func = copy.deepcopy(self.q_func)


    def get_q_hat(self, s):

        s = torch.from_numpy(s).type(torch.FloatTensor).resize(1,1,8)
        q_hat = self.q_func.forward(s)
        return q_hat

    def get_target(self, s):

        s = torch.from_numpy(s).type(torch.FloatTensor).resize(1,1,8)
        q_hat = self.target_func.forward(s)
        return q_hat


    def get_greedy_action(self, s):
        q_hat = self.get_q_hat(s)
        values, indices = torch.max(q_hat[0, 0, :], 0)
        action_idx = indices.item()
        return self.idx_action_map[action_idx]

    def update_q_value(self, s, a, r, q_max):
        self.optim.zero_grad()

        action_idx = self.action_idx_map[a]
        q_val_curr = self.get_q_hat(s)[0, 0, action_idx]
        expected_q_val = q_max*self.gamma + r

        loss = self.q_func.get_loss(q_val_curr=q_val_curr, q_val_expected=expected_q_val)

        loss.backward()
        self.optim.step()