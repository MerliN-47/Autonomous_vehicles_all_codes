#  Recursive estimation of position of vehicle along trajectory 
## using available measurements and motion model
#  Vehcile equipped with LiDar sensor returning range and bearing measurements
## corresponding to individual landmarks in environment. 
#  Global positions of the landmarks are assumedd to be known beforehand. 
## assuming known data association of which measurement belongs to which landmark

#  Motion model : recieves linear and angular velocity odometry readings as inputs,
## and outputs the state (2D pose) of vehicle
#  Measurement model : relates to current pose of the vehicle to the LiDar range 
## and bearing measurements 

# Both models are non-linear hence using extended kalman filter as the state estimator

# Unpacking available data

import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('/data.pickle','rb') as f:
    data = pickle.load(f)

# timestamps (seconds)
t = data['t'] 
# initial x position (meters)
x_init = data['x_init']
# initial y position (meters)
y_init = data['y_init']
# initial theta position (radians)
th_init = data['th_init']

# input signal
# translational velocity input (m/sec) 
v = data['v']
# rotational velocity input (rad/sec)
om = data['om']

# Bearing and range measurements , LiDar constants
# Bearing to each landmarks center in the frame attached to the laser (rad)
b = data['b']
# Range measurements 
r = data['r']
# x,y positions of landmarks (m)
l = data['l']
# Distance between robot center and laser rangefinder (m)
d = data['d']

# Parameter intialization

# translation velocity variance 
v_var = 0.01
# rotational velocity variance
om_var = 0.01
# range measurements variance 
r_var = 0.01
# bearing measurement variance
b_var = 1.0
# input noise covariance 
Q_km = np.diag([v_var , om_var])
# measurement noise covariance
cov_y = np.diag([r_var , b_var])
# estimated states x,y and theta
x_est = np.zeros([len(v),3])
# state covariance matrix 
P_est = np.zeros([len(v),3,3])
# initial state
x_est[0] = np.array([x_init , y_init , th_init]) 
# initial state covariance
P_est[0] = np.diag([1,1,0.1])

#  Wraping all estimated theta values to (-pi,pi] as orientation
## measurement should coincide with bearing measurements

def wraptopi(x):
    if x > np.pi:
        x = x - (np.floor(x / (2*np.pi)) + 1) * 2 * np.pi
    elif x < -np.pi:
        x = x + (np.floor(x / (-2*np.pi)) + 1) * 2 * np.pi
    return x

#  Correction step
#  Measurement update function takes in available landmark measurement l 
## and updates the current state estimate x_k_hat

def measurement_update(lk , rk , bk , P_check , x_check):
    x_k = x_check[0]
    y_k = x_check[1]
    theta_k = wraptopi(x_check[2])
    x_1 = lk[0]
    y_1 = lk[1]
    d_x = x_1 - x_k - d*np.cos(theta_k)
    d_y = y_1 - y_k - d*np.sin(theta_k)
    r = np.sqrt(d_x**2 + d_y**2)
    phi = np.arctan2(d_y,d_x) - theta_k

    # Computing the JACOBIAN
    H_k = np.zeros((2,3))
    H_k[0,0] = -d_x / r
    H_k[0,1] = -d_y / r
    H_k[0,2] = d*(d_x*np.sin(theta_k) - d_y*np.cos(theta_k)) / r
    H_k[1,0] = d_y / r**2
    H_k[1,1] = -d_x / r**2
    H_k[1,2] = -1-d*(d_y*np.sin(theta_k) + d_x*np.cos(theta_k)) / r**2

    M_k = np.identity(2)
    y_out = np.vstack([r , wraptopi(phi)])
    y_mes = np.vstack([rk , wraptopi(bk)])

    # Computing KALMAN GAIN
    K_k = P_check.dot(H_k.T).dot(np.linalg.inv(H_k.dot(P_check).dot(H_k.T) + M_k.dot(cov_y).dot(M_k.T)))

    # Correction of predicted state
    x_check = x_check + K_k.dot(y_mes - y_out)
    x_check[2] = wraptopi(x_check[2])

    # Correct covariance
    P_check = (np.identity(3) - K_k.dot(H_k)).dot(P_check)

    return x_check , P_check

# Prediction step 
# Filter looping 
# setting initial values 
P_check = P_est[0]
x_check = x_est[0,:].reshape(3,1)
for k in range(1,len(t)):
    delta_t = t[k] - t[k-1]
    theta = wraptopi(x_check[2])
    # Updating state with odometry readings
    F  = np.array([[np.cos(theta),0],[np.sin(theta),0],[0,1]], dtype = 'object')
    inp = np.array([[v[k-1]],[om[k-1]]])
    x_check = x_check + F.dot(inp).dot(delta_t)
    x_check[2] = wraptopi(x_check[2])
    # Motion model jacobian wrt last state
    F_km = np.zeros([3,3])
    F_km = np.array([[1,0,-np.sin(theta)*delta_t*v[k-1]],[0,1,np.cos(theta)*delta_t*v[k-1]],[0,0,1]],dtype = 'object')
    # Motion model jacobian wrt noise 
    L_km  = np.zeros([3,2])
    L_km = np.array([[np.cos(theta)*delta_t,0],[np.sin(theta)*delta_t,0],[0,1]],dtype = 'object')
    # Propagating uncertainty
    P_check = F_km.dot(P_check.dot(F_km.T)) + L_km.dot(Q_km.dot(L_km.T))
    # Update state estimate using available landmark measurement
    for i in range(len(r[k])):
        x_check , P_check = measurement_update(l[i] , r[k,i] , b[k,i] , P_check.astype(float) , x_check.astype(float))
        # final state predictions for timestep
        x_est[k,0] = x_check[0]
        x_est[k,1] = x_check[1]
        x_est[k,2] = x_check[2]
        P_est[k,:,:] = P_check

# Plotting resulting state estimates
state_fig = plt.figure()
ax = state_fig.add_subplot(111)
ax.plot(x_est[:,0] , x_est[:,1])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Estimated trajectory')
plt.show()

state_fig = plt.figure()
ax = state_fig.add_subplot(111)
ax.plot(t[:] , x_est[:,2])
ax.set_xlabel('Time [s]')
ax.set_ylabel('Theta [rad]')
ax.set_title('Estimated trajectory')
plt.show()