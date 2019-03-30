import numpy as np
import math
import pickle

class TabularQLearning(object):

    def __init__(self, n_states, min_state_val, max_state_val, gamma=.5, alpha=.1):

        self.min_state_val = min_state_val
        self.max_state_val = max_state_val
        self.n_states = n_states
        self.state_buckets = np.linspace(min_state_val, max_state_val, n_states, endpoint=False)
        self.bin_width = self.state_buckets[1] - self.state_buckets[0]
        self.actions = [.25, .2, .15, .1, .05, 0, -.25, -.2, -.15, -.1, -.05]
        #self.actions = np.linspace(-.05,.05,21)

        self.idx_action_map = dict(zip(range(len(self.actions)), self.actions))
        self.action_idx_map = dict(zip(self.actions, range(len(self.actions))))


        #self.q_table = np.random.randn(n_states, len(self.actions))
        self.q_table = np.random.uniform(-10, 0, (n_states, len(self.actions)))

        self.gamma = gamma
        self.alpha = alpha


    def get_state_idx(self, s):

        state_idx = (s - self.min_state_val) // self.bin_width
        return int(state_idx)


    def get_greedy_action(self, s):

        state_idx = self.get_state_idx(s)
        action_idx = np.argmax(self.q_table[state_idx, :])

        return self.idx_action_map[action_idx]


    def get_random_action(self):

        return np.random.choice(self.actions, 1)


    def get_epsilon_greedy_action(self, s, eps=.2):

        alpha = np.random.uniform(0,1)
        if alpha < eps:
            action = self.get_random_action()[0]
        else:
            action = self.get_greedy_action(s)

        return action

    def get_q_max(self, s):

        state_idx = self.get_state_idx(s)
        return np.max(self.q_table[state_idx, :])


    def update_q_value(self, s, a, r, q_max):
        state_idx = self.get_state_idx(s)
        action_idx = self.action_idx_map[a]

        curr_q = self.q_table[state_idx, action_idx]
        self.q_table[state_idx, action_idx] = curr_q + self.alpha*(r + self.gamma*q_max - curr_q)
        new_q = self.q_table[state_idx, action_idx]


        print("Q-value update ({:.4f}, {:.2f}): {:.4f} --> {:.4f}".format(s, a, curr_q, new_q))



    def dump_model(self):
        with open('models/q-table.p', 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_model(self, fname):
        with open(fname, 'rb') as f:
            self.q_table = pickle.load(f)