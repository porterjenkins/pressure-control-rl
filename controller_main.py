import numpy as np
import matlab.engine
from utils import reward


step_size = .2
epsilon = 1.0e-3
n_steps = 2000
frac_sec = 0.5


action_set = np.linspace(-.25, .25, 11)

for i in range(10):

    a = np.random.choice(action_set, 1)[0]
    frac_sec += a

    if frac_sec > 1 or frac_sec < 0:
        continue

    eng = matlab.engine.start_matlab()


    pressure_seq, prms = eng.sim_main(float(frac_sec), n_steps, nargout=2)
    pressure_seq = np.array(pressure_seq)
    prms = np.array(prms)
    r = reward(prms)
    print("Iteration: {}, Burn Fraction: {}, reward: {}".format(i, frac_sec, r))


eng.quit()