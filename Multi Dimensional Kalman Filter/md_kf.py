import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

# Class that implements the basic functionality of a regular kalman filter
# for a multidimensional state prediction
class kalman_filter:
    #Initialize the filter with
    # 1. initial state estimate matrix
    # 2. initial process covariance matrix
    # 3. expected error in measurement matrix
    # 4. coefficient matrix to calculate new state from old state
    # 5. process noise covariance matrix
    # 6. noise in measurement. Default = 0
    # 7. noise in estimate. Default = 0
    # 8. coefficient matrix used to select attributes measured for calculation of KG. Default = [I]
    # 9. coefficient matrix to select attributes being measured to set measurement. Default = [I]
    def __init__(self,
    init_state,                     # 1
    init_covariance,                # 2
    measurement_error,              # 3
    A,                              # 4
    process_noise = 0,              # 5
    measurement_noise = 0,          # 6
    estimate_noise = 0,             # 7
    H = None,                       # 8
    C = None):                      # 9

        self.X = init_state                # Stores state estimate
        self.P = init_covariance           # Stores process covariance
        self.R = measurement_error         # Stores measurement error for each measurement
        self.Q = process_noise             # Stores process noise covariance matrix
        self.Z = measurement_noise         # Stores noise in measurement covariance matrix
        self.A = A                         # X|k+1 = A*X|k + w
        self.w = estimate_noise            # Stores noise in estimate
        if H == None:
            self.H = np.eye(np.shape(init_state)[0])
        else:
            self.H = H                  # H is used to select attributes being measured

        if C == None:
            self.C = np.eye(np.shape(init_state)[0])
        else:
            self.C = C

    # Run one new iteration using an incoming measurement
    # 1. Predict an estimate
    # 2. Store new measurement
    # 3. Calculate Kalman Gain
    # 4. Calculate estimate for this iteration
    def step(self,measurement):
        self.predict_estimate()
        self.set_measurement(measurement)
        self.calculate_KG()
        self.calculate_estimate()

    def predict_estimate(self):
        self.X = np.matmul(self.A,self.X) + self.w
        self.P = np.matmul(np.matmul(self.A,self.P), self.A.T) + self.Q

    def set_measurement(self, meas_state):
        self.Y = np.matmul(self.C, meas_state) + self.Z

    def calculate_KG(self):
        self.KG = np.diag(np.diag(np.matmul(self.P, self.H)/(np.matmul(np.matmul(self.H, self.P), self.H.T) + self.R)))

    def calculate_estimate(self):
        self.X = self.X + np.matmul(self.KG,(self.Y - np.matmul(self.H,self.X)))
        self.P = np.diag(np.diag(np.matmul((np.eye(np.shape(self.P)[0])- np.matmul(self.KG, self.H)), self.P)))

    # Access function to get the current estimate
    def get_estimate(self):
        return (self.X, self.P)

    # Access function to get the kalman gain
    def get_KG(self):
        return self.KG


#Test code to see if the kalman filter works as expected
# 1. Set a test state value
# 2. Initialize the kalman filter
# 3. Generate a new measurement with random error each iteration
# 4. Plot the kalman gains and the estimate
#
# As the estimate becomes more accurate, the Kalman Gain should become small
#
# State Matrix = X = [x y z x' y' z' x" y" z"]'  --> Position, Velocity, Acceleration
#

if __name__ == "__main__":

    #Force matplotlib to use the tkinter library to display plots
    matplotlib.use('TkAgg')

    random.seed()

    true_state = np.array([100, 100, 110, -10, 12, -12, 0.5, 0.5, 0.5]).T     #True State

    dT = 1
    init_state = np.array([123, 112, 125, -12, 10, -15, 0, 0.3, 0.4]).T      # Initial estimate with some error
    init_covariance = np.array([[4, 3, 5, 6, 2, 3, -0.5, -0.2, -0.1]]*9)        # Error in initial estimate
    init_covariance = np.diag(np.diag(init_covariance * init_covariance.T)) # Covariance of error (non-diagonal discarded)
    measurement_error = np.array([30, 30, 30, 10, 10, 10, 0.5, 0.5, 0.5])    # Measurement Error
    measurement_error = np.diag(np.diag(measurement_error * measurement_error.T)) # Covariance of error (non-diagonal discarded)

    # For a 3D Position, Velocity, Acceleration measurement, A =
    #       1   0   0   dT  0   0   0.5*dT^2    0           0
    #       0   1   0   0   dT  0   0           0.5*dT^2    0
    #       0   0   1   0   0   dT  0           0           0.5*dT^2
    #       0   0   0   1   0   0   dT          0           0
    #       0   0   0   0   1   0   0           dT          0
    #       0   0   0   0   0   1   0           0           dT
    #       0   0   0   0   0   0   1           0           0
    #       0   0   0   0   0   0   0           1           0
    #       0   0   0   0   0   0   0           0           1
    A = np.eye(9)
    for i in range(5):
        A[i][i+3] = dT

    for i in range(3):
        A[i][i+6] = dT*dT/2

    kf = kalman_filter(init_state = init_state, init_covariance = init_covariance, measurement_error = measurement_error, A = A)

    x_estimates = np.array([])
    kg = np.array([])
    x_axis = np.array([])
    x_tv = np.array([])
    measurement_vec = np.array([])

    for i in range(60):
        measurement = true_state.copy()
        measurement[:3] = measurement[:3] + random.randint(-30,30)
        measurement[3:6] = measurement[3:6] + random.randint(-10,10)
        measurement[6:9] = measurement[6:9] + random.randint(-5,5)/10.0

        kf.step(measurement)

        x_estimates = np.append(x_estimates, kf.get_estimate()[0][0])
        kg = np.append(kg,kf.get_KG()[0][0])
        x_axis = np.append(x_axis,int(i))
        x_tv = np.append(x_tv,true_state[0])
        measurement_vec = np.append(measurement_vec,measurement[0])

        true_state[3:6] = true_state[3:6] + true_state[6:9]*dT
        true_state[:3] = true_state[:3] + dT*true_state[3:6] + 0.5*true_state[6:9]*dT*dT

        #true_state[0] = true_state[0] + dT*true_state[1]

    # Plot Kalman Gain and estimate
    plt.subplot(211)
    plt.plot(x_axis, x_tv,'g')
    plt.plot(x_axis, x_estimates, 'r--')
    plt.plot(x_axis, measurement_vec, 'b.')
    plt.subplot(212)
    plt.plot(x_axis, kg, 'b--')

    plt.show()
