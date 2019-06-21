import numpy as np
import sympy as sp
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
    # 4. method that calculates the next state from the current state and the control variable
    # 5. The function to calculate jacobian of a function
    # 6. process noise covariance matrix
    # 7. noise in measurement. Default = 0
    # 8. noise in estimate. Default = 0
    # 9. coefficient matrix used to select attributes measured for calculation of KG. Default = [I]
    # 10. coefficient matrix to select attributes being measured to set measurement. Default = [I]
    def __init__(self,
    init_state,                     # 1
    init_covariance,                # 2
    measurement_error,              # 3
    next_state,                     # 4
    jacobian,                       # 5
    process_noise = 0,              # 6
    measurement_noise = 0,          # 7
    estimate_noise = 0,             # 8
    H = None,                       # 9
    C = None):                      # 10

        self.X = init_state                # Stores state estimate
        self.P = init_covariance           # Stores process covariance
        self.R = measurement_error         # Stores measurement error for each measurement
        self.Q = process_noise             # Stores process noise covariance matrix
        self.Z = measurement_noise         # Stores noise in measurement covariance matrix
        self.w = estimate_noise            # Stores noise in estimate
        if H is None:
            self.H = np.eye(np.shape(init_state)[0])
        else:
            self.H = H                  # H is used to select attributes being measured

        if C is None:
            self.C = np.eye(np.shape(init_state)[0])
        else:
            self.C = C

        self.XnewFunc = next_state
        self.jacobian = jacobian

    # Run one new iteration using an incoming measurement
    # 1. Predict an estimate
    # 2. Store new measurement
    # 3. Calculate Kalman Gain
    # 4. Calculate estimate for this iteration
    def step_predict(self,control):
        self.predict_estimate(control)

    def step_correct(self,measurement):
        self.set_measurement(measurement)
        self.calculate_KG()
        self.calculate_estimate()

    def predict_estimate(self, control):
        self.X = self.XnewFunc(self.X, control) + self.w
        J = np.diag(np.diag(self.jacobian(self.X, control)))
        self.P = np.diag(np.diag(np.matmul(np.matmul(J,self.P), J.T))) + self.Q

    def set_measurement(self, meas_state):
        self.Y = np.matmul(self.C, meas_state) + self.Z

    def calculate_KG(self):
        self.KG = np.diag(np.diag(np.matmul(self.P, self.H.T)/(np.matmul(np.matmul(self.H, self.P), self.H.T) + self.R)))

    def calculate_estimate(self):
        self.X = self.X + np.matmul(self.KG,(self.Y - np.matmul(self.H,self.X)))
        self.P = np.diag(np.diag(np.matmul((np.eye(np.shape(self.P)[0]) - np.matmul(self.KG, self.H)), self.P)))

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
    def new_state(current_state, acceleration, step=1):
        next_state = np.zeros(6)
        next_state[:3] = current_state[:3] + current_state[3:6]*step + 0.5*acceleration*step*step
        next_state[3:6] = current_state[3:6] + acceleration*step
        return next_state

    def jacobian(state, acceleration):
        J = np.zeros((6,6))
        J[:3, :3] = 1 + acceleration/state[3:6]
        J[3:6, :3] = acceleration/state[3:6]
        J[:3, 3:6] = state[3:6]/acceleration + 1
        J[3:6, 3:6] = 1
        return J


    #Force matplotlib to use the tkinter library to display plots
    matplotlib.use('TkAgg')

    random.seed()

    true_state = np.array([100, 100, 110, -10, 12, -12])     #True State
    acceleration = np.array([0.5, 0.5, 0.5])

    init_state = np.array([123, 112, 125, -12, 10, -15])    # Initial estimate with some error
    init_control = np.array([0.5, 0.5, 0.5])                  # Initial estimate of control variable acceleration value
    init_covariance = np.array([[4, 3, 5, 6, 2, 3]]*6)          # Error in initial estimate
    init_covariance = np.diag(np.diag(init_covariance * init_covariance.T)) # Covariance of error (non-diagonal discarded)
    measurement_error = np.array([[30, 30, 30, 10, 10, 10]]*6)
    measurement_error = np.diag(np.diag(measurement_error * measurement_error.T)) # Covariance of error (non-diagonal discarded)

    kf = kalman_filter(init_state = init_state, init_covariance = init_covariance, measurement_error = measurement_error, next_state = new_state, jacobian = jacobian)

    x_estimates = np.array([])
    kg = np.array([])
    x_axis = np.array([])
    x_tv = np.array([])
    measurement_vec = np.array([])
    loss = np.array([])

    for i in range(50):
        measurement = true_state.copy()
        acceleration_meas = acceleration.copy()
        measurement[:3] = measurement[:3] + random.randint(-30,30)
        measurement[3:6] = measurement[3:6] + random.randint(-10,10)
        acceleration_meas = acceleration_meas + random.randint(-3,3)/10

        kf.step_predict(acceleration_meas)
        x_estimates = np.append(x_estimates, kf.get_estimate()[0][0])

        kf.step_correct(measurement)

        kg = np.append(kg,kf.get_KG()[0][0])
        x_axis = np.append(x_axis,int(i))
        x_tv = np.append(x_tv,true_state[0])
        measurement_vec = np.append(measurement_vec,measurement[0])

        loss = np.append(loss,abs(x_estimates[i] - true_state[0]))

        true_state = new_state(true_state,acceleration)

        #true_state[0] = true_state[0] + dT*true_state[1]

    # Plot Kalman Gain and estimate
    fig = plt.figure()
    a1 = fig.add_subplot(221)
    a1.plot(x_axis, x_tv,'g')
    a1.plot(x_axis, x_estimates, 'r--')
    a1.plot(x_axis, measurement_vec, 'b.')
    a2 = fig.add_subplot(222)
    a2.plot(x_axis, kg, 'b--')
    a3 = fig.add_subplot(223)
    a3.plot(x_axis, loss, 'b--')
    a1.title.set_text('Prediction')
    a2.title.set_text('Kalman Gain')
    a3.title.set_text('Error')
    plt.show()
