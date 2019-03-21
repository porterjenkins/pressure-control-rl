import numpy as np
import math

class TabularQLearning(object):

    def __init__(self, n_states, min_state_val, max_state_val):

        self.min_state_val = min_state_val
        self.max_state_val = max_state_val
        self.n_states = n_states
        self.state_buckets = np.linspace(min_state_val, max_state_val, n_states, endpoint=False)
        self.bin_width = self.state_buckets[1] - self.state_buckets[0]
        self.actions = [.25, .2, .15, .1, .05, 0, -.25, -.2, -.15, -.1, -.05]

        self.action_idx_map = dict(zip(range(len(self.actions)), self.actions))


        self.q_table = np.random.randn(n_states, len(self.actions))


    def get_state_idx(self, s):

        state_idx = (s - self.min_state_val) // self.bin_width
        return int(state_idx)


    def get_greedy_action(self, s):

        state_idx = self.get_state_idx(s)
        action_idx = np.argmax(self.q_table[state_idx, :])

        return self.action_idx_map[action_idx]


    def get_random_action(self):

        return np.random.choice(self.actions, 1)



    def get_epsilon_greedy_action(self, s, eps=.2):

        alpha = np.random.uniform(0,1)
        if alpha < eps:
            action = self.get_random_action()[0]
        else:
            action = self.get_greedy_action(s)

        return action

    def quit(self):
        self.eng.quit()
