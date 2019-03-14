import matlab.engine
eng = matlab.engine.start_matlab()

# test generic matlab interace

y=eng.asin(1.)
print(y)


# test matlab function
z = eng.test_f(100.)
print(z)


eng.quit()
