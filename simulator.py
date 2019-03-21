import numpy as np
import matlab.engine

class Simulator(object):

    def __init__(self, totalsteps):

        self.totalsteps = totalsteps
        self.total_time = 30 # Total time
        self.mass_in = 30.0 # Inlet Mass Flow of Air (g/s)
        self.phi_primary = 1.0 # Phi of Primary burner
        self.frac_sec = 0.5

        # Geometry Variables

        self.pos_primary = 0.20
        self.pos_secondary = 0.8
        self.pos_ignition = 0.215

        # Acoustic Variables

        self.damp_coeff = 0.0

        # Control Variables

        self.threshold = 80
        self.control_freq = 20
        self.rept = 500 # Reporting Interval
        self.kp = 0.10

        self.eng = matlab.engine.start_matlab()

        # Setup MATLAB global vars
        self.eng.Setup_BC(self.total_time, self.mass_in, self.phi_primary, self.frac_sec, nargout=0)
        self.eng.Setup_Geometry(self.pos_primary, self.pos_secondary, self.pos_ignition, nargout=0)
        self.eng.Setup_Chemistry(nargout=0)
        self.eng.Setup_Acoustic(self.damp_coeff, nargout=0)
        self.eng.Initialize_Solution(nargout=0)



    def main(self):

        cntr = 0
        for i in range(self.rept, self.totalsteps+self.rept, self.rept):
            print("Iteration: {}".format(i))
            pressure_seq = self.eng.Time_Solver(self.rept, i, self.mass_in, self.phi_primary, self.frac_sec)

            pressure_seq = np.array(pressure_seq)
            print(pressure_seq)

            cntr += 1

    def terminate_simulator(self):
        self.eng.quit()



if __name__ == "__main__":
    sim = Simulator(totalsteps=2500)
    sim.main()
    sim.terminate_simulator()