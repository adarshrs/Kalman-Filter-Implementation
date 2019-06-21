# Kalman-Filter-Implementation
A Python3.6 based implementation of the Kalman Filter

1. Single Variable Kalman Filter:
	Filter measurements to predict for a single constant variable
2. Multi Dimensional Kalman Filter:
	Filter measurements for multiple variables to predict the state of a system
3. Extended Kalman Filter:
	Implementation of the Extended Kalman Filter for non-linear systems
4. Unscented Kalman Filter:
	Implementation of the Unscented Kalman Filter for non-linear systems

Code includes sample test case that can be run. To use with a different system, import file and use the kalman_filter class.

Next state function (and Jacobian Function in EKF) are to be provided at the time of initialization.

Call the step_predict and step_correct functions in each iteration to do a prediction and measurement correction of the system.
