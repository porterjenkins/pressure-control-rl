import numpy as np
import matlab.engine

eng = matlab.engine.start_matlab()

pressure_seq, prms = eng.sim_main(0.0, 1000, nargout=2)
pressure_seq = np.array(pressure_seq)
prms = np.array(prms)

eng.quit()