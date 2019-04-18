import numpy as np
import matlab.engine
from utils import *
from q_learning import TabularQLearning, LinearQLearning, LstmQLearning, MlpQlearning
import matplotlib.pyplot as plt
from memory_replay import ReplayMemory


class Simulator(object):
    """
    Simulator base class
    """

    def __init__(self, controller, totalsteps, target, state_size, persist):
        """

        :param controller:
        :param totalsteps:
        :param target:
        :param state_size:
                    -Tabular: number of state buckets
                    -Linear: number of previous prms values to consider as state features
        :param persist:
        """

        self.totalsteps = totalsteps
        self.target = target # reward target
        self.persist = persist
        self.controller_type = controller

        # init controller
        assert controller in ['tabular', 'linear', 'mlp', 'lstm']
        if controller == 'tabular':
            self.controller = TabularQLearning(n_states=state_size, min_state_val=0, max_state_val=5.0e3, gamma=.1)
        elif controller == 'linear':
            self.controller = LinearQLearning(n_features=state_size)
        elif controller == 'mlp':
            self.controller = MlpQlearning(n_features=state_size)
        elif controller == 'lstm':
            self.controller = LstmQLearning(seq_size=state_size, hidden_dim=16)

    def init_matlab_env(self):

        self.total_time = 30  # Total time
        self.mass_in = 140.0  # Inlet Mass Flow of Air (g/s)
        self.phi_primary = 1.0  # Phi of Primary burner
        self.frac_sec = 0.0

        # Geometry Variables

        self.pos_primary = 0.20
        self.pos_secondary = 0.75
        self.pos_ignition = 0.215

        # Acoustic Variables

        self.damp_coeff = 0.0008

        # Control Variables

        #self.threshold = 100
        self.control_freq = 20
        self.rept = 500  # Reporting Interval
        self.kp = 1e-3

        self.eng = matlab.engine.start_matlab()

        # Setup MATLAB global vars
        self.eng.Setup_BC(self.total_time, self.mass_in, self.phi_primary, self.frac_sec, nargout=0)
        self.eng.Setup_Geometry(self.pos_primary, self.pos_secondary, self.pos_ignition, nargout=0)
        self.eng.Setup_Chemistry(nargout=0)
        self.eng.Setup_Acoustic(self.damp_coeff, nargout=0)
        self.eng.Initialize_Solution(nargout=0)

    def shutdown_matlab_env(self):
        self.eng.quit()
        self.frac_sec = 0.5



    def main(self, n_episodes):

        output = {'settling time': [],
                  'mae': []
                  }
        memory = ReplayMemory(10000)


        for episode_cnt in range(n_episodes):
            print("STARTING EPISODE: {}".format(episode_cnt+1))
            self.init_matlab_env()
            p = np.zeros(self.totalsteps)
            prms = []

            cntr = 0
            try:
                for i in range(self.rept, self.totalsteps+self.rept, self.rept):
                    start_idx = cntr * self.rept
                    end_idx = (cntr + 1) * self.rept

                    if cntr == 0:
                        # initialize state
                        p_i = self.eng.Time_Solver(self.rept, i, self.mass_in, self.phi_primary, self.frac_sec)
                        p_i = np.array(p_i)
                        prms_i = rms(p_i)
                        prms.append(prms_i)

                        p[start_idx:end_idx] = p_i
                        #state_i = prms_i
                        cntr += 1
                        continue

                    state_i_features = self.controller.feature_extractor(prms)
                    action_i = self.controller.get_epsilon_greedy_action(state_i_features, eps=.1)
                    self.update_frac_sec(action_i)

                    p_i = self.eng.Time_Solver(self.rept, i, self.mass_in, self.phi_primary, self.frac_sec)
                    p_i = np.array(p_i)
                    prms_i = rms(p_i)
                    prms.append(prms_i)
                    p[start_idx:end_idx] = p_i

                    reward_i = reward(prms_i, target=self.target)

                    state_prime_features = self.controller.feature_extractor(prms)
                    print("Iteration: {}, PRMS: {:.4f}, action: {} reward: {:.4f}".format(cntr + 1, prms_i, action_i,
                                                                                          reward_i))

                    q_max_next = self.controller.get_q_max(state_prime_features)

                    self.controller.update_q_value(state_i_features, action_i, reward_i, q_max_next)


                    cntr += 1

                self.shutdown_matlab_env()
                #self.plot(p, fname='pressure-{}.pdf'.format(episode_cnt+1))
                self.plot_prms(prms, fname='prms-{}-{}.pdf'.format(episode_cnt+1, self.controller_type))

                output['settling time'].append(self.get_settle_time(prms, self.target))
                output['mae'].append(self.get_mae(prms, self.target))


            except matlab.engine.MatlabExecutionError as err:
                print("Error in matlab execution")
                pass

        if self.persist:
            self.controller.dump_model(fname='models/{}.p'.format(self.controller_type))

        return output

    def get_settle_time(self, prms, target):
        eps = .05
        sequence_rule = 4

        prms_arr = np.array(prms, dtype=np.float32)
        err = (prms_arr - target) / target

        time_cntr = 0
        consecutive_cntr = 0

        for i in err:
            if i < eps or i < -eps:
                consecutive_cntr += 1
            else:
                time_cntr +=1

            if consecutive_cntr == 4:
                break

        return time_cntr


    def get_mae(self, prms, target):
        err = np.abs(prms[-1] - target)
        return err


    def update_frac_sec(self, action):
        self.frac_sec += float(action)


    def plot_prms(self, arr, fname):
        y_label = 'Prms (Pa)'
        x_label = 'time (sec)'


        x = np.arange(1, len(arr)+1)*self.rept*1.0e-4
        plt.plot(x, arr)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.savefig("figs/" + fname)
        plt.close()
        plt.clf()





if __name__ == "__main__":

    controller = input("Specify Q() function [linear, tabular, mlp, lstm]: ")
    state_size = int(input("Specify state size:"))

    sim = Simulator(controller=controller, totalsteps=50000, target=300, state_size=8, persist=True)
    #sim.controller.load_model('models/q-table.p')
    # train
    output = sim.main(n_episodes=100)

    print(output)
