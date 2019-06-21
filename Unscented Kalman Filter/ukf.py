import numpy as np
import scipy.linalg
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
    # 5. process noise covariance matrix
    # 6. Tuning parameter that decides how much spread in sigma points. Order of 10e-3
    # 7. Tuning parameter that gives information on the type of noise in the system. Default = 2 for gaussian
    # 8. noise in measurement. Default = 0
    # 9. noise in estimate. Default = 0
    # 10. coefficient matrix used to select attributes measured for calculation of KG. Default = [I]
    # 11. coefficient matrix to select attributes being measured to set measurement. Default = [I]
    # 12. scaling parameter to decide spread of sigma points
    def __init__(self,
    init_state,                     # 1
    init_covariance,                # 2
    measurement_error,              # 3
    next_state,                     # 4
    alpha=0.001,                     # 5
    beta=2,                         # 6
    process_noise = 0,              # 7
    measurement_noise = 0,          # 8
    estimate_noise = 0,             # 9
    H = None,                       # 10
    C = None,                       # 11
    K=0):                           # 12

        self.X = init_state                # Stores state estimate
        self.P = init_covariance           # Stores process covariance
        self.R = measurement_error         # Stores measurement error for each measurement
        self.Q = process_noise             # Stores process noise covariance matrix
        self.Z = measurement_noise         # Stores noise in measurement covariance matrix
        self.w = estimate_noise            # Stores noise in estimate
        if H == None:
            self.H = np.eye(np.shape(init_state)[0])
        else:
            self.H = H                  # H is used to select attributes being measured

        if C == None:
            self.C = np.eye(np.shape(init_state)[0])
        else:
            self.C = C

        self.K = K

        self.Py = np.zeros(np.shape(self.X))
        self.Py = np.diag(np.square(self.Py))

        self.XnewFunc = next_state

        self.N = len(self.X)
        self.alpha = alpha
        if beta is None:
            self.beta = 1-self.alpha**2
        else:
            self.beta = beta

        self.lmbda = (self.alpha**2)*(self.N + self.K) - self.N
        self.gamma = np.sqrt(self.N + self.lmbda)

        self.wm = np.array([1/(2*self.N + 2*self.lmbda)]*(2*self.N+1))
        self.wc = np.array([1/(2*self.N + 2*self.lmbda)]*(2*self.N+1))
        self.wc[0] = self.lmbda/(self.N + self.lmbda) + (1 - self.alpha**2 + self.beta)
        self.wm[0] = self.lmbda/(self.N + self.lmbda)

    # Run one new iteration using an incoming measurement
    # 1. Predict an estimate
    # 2. Store new measurement
    # 3. Calculate Kalman Gain
    # 4. Calculate estimate for this iteration
    def step_predict(self,control):
        self.predict_estimate(control)

    def step_correct(self,measurement):
        self.set_measurement_error(measurement)
        self.calculate_KG()
        self.calculate_estimate()

    def calculate_sigma_X(self):
        # Calculate sigma points
        self.sigma = np.zeros((self.N,2*self.N+1))
        Psqrt = np.sqrt(abs(self.P))

        # Positive direction
        self.sigma[:,0] = np.reshape(self.X,(self.N,))
        for i in range(1,self.N):
            self.sigma[:,i] = np.reshape(np.reshape(self.X,(self.N,1)) + np.reshape(self.gamma * Psqrt[:, i],(self.N,1)), (self.N,))

        # Negative direction
        for i in range(self.N,2*self.N):
            self.sigma[:,i] = np.reshape(np.reshape(self.X,(self.N,1)) - np.reshape(self.gamma * Psqrt[:, i-self.N],(self.N,1)), (self.N,))

        '''fig = plt.figure()
        a1 = fig.add_subplot(111)
        a1.plot(self.sigma[0,:],self.sigma[0,:],'g.')
        a1.plot(self.X[0],self.X[0],'r.')
        plt.show()
        input()'''

    def calculate_sigma_observations(self):
        sigma = np.zeros((self.N,2*self.N+1))

        for i in range(2*self.N+1):
            sigma[:, i] = np.matmul(self.H, sigma[:, i])

        return sigma

    def calculate_observation_variance(self,zb,z_sigma):
        d = (z_sigma.T - zb.T).T
        P = self.R

        for i in range(self.N*2+1):
            P = P + np.matmul(self.wc[i] * d[:,i], d[:,i].T)
        return np.diag(np.diag(P))

    def calculate_observation_covariance(self,zb,z_sigma):
        dx = (self.sigma.T - self.X.T).T
        dz = (z_sigma.T - zb.T).T
        P = np.zeros((dx.shape[0], dz.shape[0]))

        for i in range(self.N):
            P = P + np.matmul(self.wc[i] * dx[:,i], dz[:,i].T)

        P = np.diag(np.diag(P))
        return P

    def predict_estimate(self, control):
        self.calculate_sigma_X()

        newX = np.zeros(np.shape(self.sigma))

        for i in range(self.N*2+1):
            newX[:,i] = self.XnewFunc(self.sigma[:,i], control)

        self.X = np.matmul(self.wm,newX.T).T

        d = (newX.T - self.X.T).T
        self.P = np.zeros((self.N,self.N))
        for i in range(self.N*2+1):
            self.P = self.P + np.matmul(self.wc[i] * d[:,i],d[:,i].T)

        self.P += self.Q

    def set_measurement_error(self, meas_state):
        self.Y = meas_state - self.H @ np.reshape(self.X,(self.N,1))

        self.calculate_sigma_X()
        zb = np.matmul(self.wm, self.sigma.T).T
        z_sigma = self.calculate_sigma_observations()
        self.st = self.calculate_observation_variance(zb, z_sigma)
        self.pxz = self.calculate_observation_covariance(zb,z_sigma)

    def calculate_KG(self):
        self.KG = np.matmul(self.pxz, np.linalg.inv(self.st))

    def calculate_estimate(self):
        self.X = np.reshape(self.X,(self.N,1)) + np.matmul(self.KG,self.Y)
        self.P = self.P - np.matmul(np.matmul(self.KG, self.st), self.KG.T)

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
# 4. Plot the kalman gains and the wmestimate
#
# As the estimate becomes more accurate, the Kalman Gain should become small
#
# State Matrix = X = [x y z x' y' z' x" y" z"]'  --> Position, Velocity, Acceleration
#

if __name__ == "__main__":
    def new_state(current_state, acceleration, step=1):
        next_state = np.zeros(np.shape(current_state))
        next_state[:3] = (current_state[:3].T + current_state[3:6].T*step + 0.5*acceleration.T*step*step).T
        next_state[3:6] = (current_state[3:6].T + acceleration.T*step).T
        return next_state


    #Force matplotlib to use the tkinter library to display plots
    matplotlib.use('TkAgg')

    random.seed()

    true_state = np.array([100, 100, 110, -10, 12, -12])     #True State
    acceleration = np.array([0.5, 0.5, 0.5])

    init_state = np.array([123, 112, 125, -12, 10, -15])    # Initial estimate with some error
    init_control = np.array([0.5, 0.5, 0.5])                  # Initial estimate of control variable acceleration value
    init_covariance = np.diag([5, 5, 5, 5, 5, 5])**2          # Error in initial estimate
    measurement_error = np.diag([30, 30, 30, 10, 10, 10])**2

    kf = kalman_filter(init_state = init_state, init_covariance = init_covariance, measurement_error = measurement_error, next_state = new_state)

    x_estimates = np.array([])
    kg = np.array([])
    x_axis = np.array([])
    x_tv = np.array([])
    measurement_vec = np.array([])
    loss = np.array([])
    covariance = np.array([])

    for i in range(50):
        measurement = true_state.copy()
        acceleration_meas = acceleration.copy()
        measurement = np.reshape(measurement,(6,1)) + np.matmul(np.sqrt(measurement_error),np.random.randn(6,1))
        acceleration_meas = acceleration_meas + random.randint(-1,1)/10

        kf.step_predict(acceleration_meas)
        x_estimates = np.append(x_estimates, kf.get_estimate()[0][0])

        kf.step_correct(measurement)

        kg = np.append(kg,kf.get_KG()[0][0])
        x_axis = np.append(x_axis,int(i))
        x_tv = np.append(x_tv,true_state[0])
        measurement_vec = np.append(measurement_vec,measurement[0])
        covariance = np.append(covariance,kf.get_estimate()[1][0][0])

        loss = np.append(loss,abs(x_estimates[i] - true_state[0]))

        #print(x_estimates[i])
        #input()

        true_state = new_state(true_state,acceleration)


    # Plot Kalman Gain and estimate
    fig = plt.figure()

    a1 = fig.add_subplot(322)
    a1.title.set_text('Prediction')
    #a1.set_ylim(-500,500)
    a1.plot(x_axis, x_tv,'g',label='True Value')
    a1.plot(x_axis, x_estimates, 'r--',label='Prediction')
    a1.plot(x_axis, measurement_vec, 'b.',label='Noisy measurement')
    a1.legend(loc='best')

    a2 = fig.add_subplot(323)
    a2.title.set_text('Kalman Gain for X position')
    a2.plot(x_axis, kg, 'b--')

    a3 = fig.add_subplot(324)
    a3.title.set_text('Error between prediction and true state')
    a3.plot(x_axis, loss, 'b--')

    a4 = fig.add_subplot(321)
    a4.title.set_text('X position true state vs time')
    a4.plot(x_axis, x_tv,'g')

    a5 = fig.add_subplot(325)
    a5.title.set_text('Covariance of X position')
    a5.plot(x_axis,covariance,'b--')
    plt.show()
