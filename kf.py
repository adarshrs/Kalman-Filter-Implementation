import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

# Class that implements the basic functionality of a regular kalman filter
class kalman_filter:
    #Initialize the filter with
    # 1. initial estimate
    # 2. initial error in estimate
    # 3. initial measurement
    # 4. expected error in measurement
    def __init__(self,init_est, init_est_error, init_meas, init_meas_error):
        self.estimate = init_est
        self.estimate_error = init_est_error
        self.measurement = init_meas
        self.measurement_error = init_meas_error

    # Start by running one iteration using the initial supplied estimates
    def start(self):
        self.calculate_KG()
        self.calculate_estimate()
        self.calculate_estimate_error()

    # Run one new iteration using an incoming measurement
    # 1. Calculate the kalman gain
    # 2. Store the new measurement
    # 3. Calculate the estimate for this iteration
    # 4. Calculate the new error in the estimate using previous estimate
    def step(self,measurement):
        self.calculate_KG()
        self.set_measurement(measurement)
        self.calculate_estimate()
        self.calculate_estimate_error()

    def calculate_KG(self):
        self.KG = self.estimate_error/(self.estimate_error + self.measurement_error)

    def set_measurement(self, meas_val):
        self.measurement = meas_val

    def calculate_estimate(self):
        self.estimate = self.estimate + self.KG*(self.measurement - self.estimate)

    def calculate_estimate_error(self):
        self.estimate_error = (1-self.KG)*self.estimate_error

    # Access function to get the current estimate
    def get_estimate(self):
        return self.estimate

    # Access function to get the kalman gain
    def get_KG(self):
        return self.KG


#Test code to see if the kalman filter works as expected
# 1. Set a test true value of 22.5
# 2. Initialize the kalman filter with
#           Estimated Value = 20
#           Estimate Error = 5
#           Measured Value = 23
#           Error in measurement = 4
# 3. Start the kalman filter and generate a new measurement with
#    some random error between -4 and 4 for each iteration
# 4. Plot the kalman gains and the estimate
#
# As the estimate becomes more accurate, the Kalman Gain should become small
if __name__ == "__main__":

    #Force matplotlib to use the tkinter library to display plots
    matplotlib.use('TkAgg')

    random.seed()

    true_value = 22.5
    kf = kalman_filter(20,5,23,4)

    kf.start()

    estimates = np.array([])
    kg = np.array([])
    x_axis = np.array([])
    tv = np.array([])

    estimates = np.append(estimates, kf.get_estimate())
    kg = np.append(kg,kf.get_KG())
    x_axis = np.append(x_axis,0)
    tv = np.append(tv,22.5)

    for i in range(200):
        measurement = 22.5 + random.randint(-40,40)/10
        kf.step(measurement)

        estimates = np.append(estimates, kf.get_estimate())
        kg = np.append(kg,kf.get_KG())
        x_axis = np.append(x_axis,int(i))
        tv = np.append(tv,true_value)


    # Plot Kalman Gain and estimate 
    plt.subplot(211)
    plt.plot(x_axis, tv,'g')
    plt.plot(x_axis, estimates, 'r--')
    plt.subplot(212)
    plt.plot(x_axis, kg, 'b--')

    plt.show()
