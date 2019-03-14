import matlab.engine
eng = matlab.engine.start_matlab()

# test rijke tube simulator
eng.Reduced_Non_Premixed_Heat_2cells_v7_Modular(nargout=0)

eng.quit()
