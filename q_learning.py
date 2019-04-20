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

    def get_feasible_actions(self, curr_p):
        """
        Restrict actions to only feasible actions given current pressure
        :param curr_p: Current pressure value
        :return:
        """

        feasible_actions = {}
        for a, idx in self.action_idx_map.items():
            if a + curr_p <= 1.0 and a + curr_p >= 0.0:

                feasible_actions[a] = idx

        return feasible_actions


    def feature_extractor(self, observation):
        pass

    def get_q_hat(self, s):
        pass
    def get_target(self, s):
        pass

    def get_greedy_action(self,s, feasible_a):
        pass

    def get_random_action(self, feasible_a):
        a_list = list(feasible_a.keys())
        #action = np.random.choice(self.actions, 1)[0]
        action = np.random.choice(a_list, 1)[0]
        action_idx = feasible_a[action]
        return action, action_idx


    def get_epsilon_greedy_action(self, s, p, eps=.2):

        alpha = np.random.uniform(0,1)
        feasible_a = self.get_feasible_actions(curr_p=p)

        if alpha < eps:
            action, a_idx = self.get_random_action(feasible_a)
        else:
            action, a_idx = self.get_greedy_action(s, feasible_a)

        return action, a_idx


    def dump_model(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self.q_func, f)

    def load_model(self, fname):
        with open(fname, 'rb') as f:
            self.q_func = pickle.load(f)

    def get_q_max(self, s, dim=None):
        """
        Action that maximized expected reward of target function
        :param s:
        :return:
        """
        q_hat = self.get_target(s)
        if isinstance(q_hat, np.ndarray):
            return np.max(q_hat)
        else:
            return q_hat.max(dim)

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

    def get_greedy_action(self, s, feasible_a):
        q_hat = self.get_q_hat(s)
        action_idx = np.argmax(q_hat)
        return self.idx_action_map[action_idx], action_idx




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
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).type(torch.FloatTensor)
        q_hat = self.target_func.forward(s)
        return q_hat

    def get_memory_batch(self):
        transitions = memory_replay.memory.sample(memory_replay.BATCH_SIZE)
        batch = memory_replay.Transition(*zip(*transitions))

        state_batch = np.concatenate(batch.state, axis=0)
        reward_batch = np.concatenate(batch.reward)
        action_batch = np.concatenate(batch.action)
        next_state_batch = np.concatenate(batch.next_state, axis=0)

        action_batch = torch.from_numpy(action_batch).type(torch.LongTensor)
        action_batch.resize_(memory_replay.BATCH_SIZE, 1)

        reward_batch = torch.from_numpy(reward_batch).type(torch.FloatTensor)


        return state_batch, reward_batch, action_batch, next_state_batch

    def update_q_value(self, s, a, r, q_max=None):
        if len(memory_replay.memory) < memory_replay.BATCH_SIZE:
            return

        state_batch, reward_batch, action_batch, next_state_batch = self.get_memory_batch()

        # q_value from policy network

        #action_idx = self.action_idx_map[a]
        q_val_curr = self.get_q_hat(state_batch).gather(1, action_batch)


        # q_value from target network
        q_max = self.get_q_max(next_state_batch, dim=1)[0]
        expected_q_val = q_max*self.gamma + reward_batch

        self.optim.zero_grad()
        loss = self.q_func.get_loss(q_val_curr=q_val_curr, q_val_expected=expected_q_val)

        loss.backward()
        self.optim.step()

    def get_greedy_action(self, s, feasible_a):
        q_hat = self.get_q_hat(s)
        a_set = np.array(list(feasible_a.values()))
        feasible_a_idx = torch.from_numpy(a_set).type(torch.LongTensor)
        feasible_q = q_hat.gather(0, feasible_a_idx)


        max_feasible_idx = feasible_q.max(0)[1].item()
        action_idx = a_set[max_feasible_idx]

        return self.idx_action_map[action_idx], action_idx

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
        if isinstance(s, np.ndarray):
            if s.ndim > 1:
                b = s.shape[0]
            else:
                b = 1
            s = torch.from_numpy(s).type(torch.FloatTensor).resize(b,1,self.n_features)
        q_hat = self.q_func.forward(s)
        return q_hat

    def get_target(self, s):
        if isinstance(s, np.ndarray):
            if s.ndim > 1:
                b = s.shape[0]
            else:
                b = 1
            s = torch.from_numpy(s).type(torch.FloatTensor).resize(b,1,self.n_features)
        q_hat = self.target_func.forward(s)
        return q_hat


    def get_greedy_action(self, s, feasible_a):

        q_hat = self.get_q_hat(s)
        a_set = np.array(list(feasible_a.values()))
        feasible_a_idx = torch.from_numpy(a_set).type(torch.LongTensor)
        feasible_a_idx.resize_(1, 1, len(a_set))

        feasible_q = q_hat.gather(2, feasible_a_idx)
        vals, idx = feasible_q.max(2)
        max_feasible_idx = idx.item()

        action_idx = a_set[max_feasible_idx]

        return self.idx_action_map[action_idx], action_idx


    def update_q_value(self, s, a, r, q_max=None):
        if len(memory_replay.memory) < memory_replay.BATCH_SIZE:
            return

        state_batch, reward_batch, action_batch, next_state_batch = self.get_memory_batch()
        action_batch.resize_(memory_replay.BATCH_SIZE, 1, 1)
        reward_batch.resize_(memory_replay.BATCH_SIZE, 1)

        # q_value from policy network

        #action_idx = self.action_idx_map[a]
        q_val_curr = self.get_q_hat(state_batch).gather(2, action_batch)


        # q_value from target network
        q_max = self.get_q_max(next_state_batch, dim=2)[0]
        expected_q_val = q_max*self.gamma + reward_batch

        self.optim.zero_grad()
        loss = self.q_func.get_loss(q_val_curr=q_val_curr, q_val_expected=expected_q_val)

        loss.backward()
        self.optim.step()