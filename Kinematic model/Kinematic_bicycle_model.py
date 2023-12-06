# Kinematic bicycle model 
# Input is velocity and steering rate 

# Kinematic equations 
# 1) xc_dot = v * cos(theta + beta)
# 2) yc_dot = v * sin(theta + beta)
# 3) theta_dot = (v * cos(beta) * tan(delta)) / L
# 4) delta_dot = w 
# 5) beta = arctan((lr * tan(delta)) / L)

# v -> bicycle speed 
# w -> steering angle rate
# delta -> steering angle 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Bicycle():
    def __init__(self):
        self.xc = 0
        self.yc = 0
        self.theta = 0
        self.delta = 0
        self.beta = 0 
        self.L = 2
        self.lr = 1.2
        self.w_max = 1.22
        self.sample_time = 0.01
    
    def reset(self):
        self.xc = 0
        self.yc = 0
        self.theta = 0
        self.delta = 0
        self.beta = 0 

# sample time for numerical integration is required and set to 10 miliseconds
# The step function takes v and w as input and updates our previously mentioned state variables
# Maximum turning rate is 1.22 rad/sec with wheelbase of 2 meters and length of 1.2 meters to its CoM from rear axle

class Bicycle(Bicycle):
    def step(self , v , w):
        self.xc = self.xc + v * np.cos(self.theta + self.beta) * self.sample_time
        self.yc = self.yc + v * np.sin(self.theta + self.beta) * self.sample_time
        self.theta = self.theta + ((v * np.cos(self.beta) * np.tan(self.delta)/ self.L)) * self.sample_time
        self.delta = w * self.sample_time + self.delta
        self.beta = np.arctan((self.lr / self.L) * np.tan(self.delta))

# Model is now defined 
# setting up different inputs

# e.g. Travel radius of 10 meters in 20 seconds

# tan(delta) = L / r
# arctan(2/10) = 0.1974
# velocity = d / t -> 2*pi*10 / 20 -> pi

# Circle trajectory
sample_time = 0.01
time_end = 20
model = Bicycle()

# Setting delta directly
model.delta = np.arctan(2/10)
t_data = np.arange(0 , time_end , sample_time)
x_data = np.zeros_like(t_data)
y_data = np.zeros_like(t_data)

for i in range(t_data.shape[0]):
    x_data[i] = model.xc
    y_data[i] = model.yc
    model.step(np.pi , 0)
    model.beta = 0
    #print('xc = ',model.xc)
    #print('yc = ',model.yc)
    #print('theta = ',model.theta)
    #print('delta = ',model.delta)
    #print('beta = ',model.beta)

plt.axis('equal')
plt.plot(x_data , y_data , label = 'Model trajectory')
plt.legend()
plt.show()

# Radius of above circle will be  10 meters, if beta is not manually set to zero, sideslip effect can be noticed
# steering angle cannot be dirctly set and must be changed through angular rate input (w) 

model.reset()
t_data = np.arange(0,time_end,sample_time)
x_data = np.zeros_like(t_data)
y_data = np.zeros_like(t_data)

for i in range(t_data.shape[0]):
    x_data[i] = model.xc
    y_data[i] = model.yc
    
    if model.delta < np.arctan(2/10):
        model.step(np.pi , model.w_max)
    else:
        model.step(np.pi , 0)
        
plt.axis('equal')
plt.plot(x_data , y_data , label = 'Model trajectory')
plt.legend()
plt.show()

#square trajectory
time_end = 60
model.reset()

t_data = np.arange(0,time_end,sample_time)
x_data = np.zeros_like(t_data)
y_data = np.zeros_like(t_data)

# maintain velocity at 4 m/s
v_data = np.zeros_like(t_data)
v_data[:] = 4

w_data = np.zeros_like(t_data)

w_data[670:670+100] = 0.753
w_data[670+100:670+100*2] = -0.753
w_data[2210:2210+100] = 0.753
w_data[2210+100:2210+100*2] = -0.753
w_data[3670:3670+100] = 0.753
w_data[3670+100:3670+100*2] = -0.753
w_data[5220:5220+100] = 0.753
w_data[5220+100:5220+100*2] = -0.753

for i in range(t_data.shape[0]):
    x_data[i] = model.xc
    y_data[i] = model.yc
    model.step(v_data[i],w_data[i])

plt.axis('equal')
plt.plot(x_data , y_data , label = 'Model trajectory')
plt.legend()
plt.show()

# Spiral path : high positive w and small negative w
time_end = 60
model.reset()

t_data = np.arange(0,time_end,sample_time)
x_data = np.zeros_like(t_data)
y_data = np.zeros_like(t_data)

# maintain velocity at 4 m/s
v_data = np.zeros_like(t_data)
v_data[:] = 4
w_data [:] = -1/100
w_data[0:100] = 1


for i in range(t_data.shape[0]):
    x_data[i] = model.xc
    y_data[i] = model.yc
    model.step(v_data[i],w_data[i])

plt.axis('equal')
plt.plot(x_data , y_data , label = 'Model trajectory')
plt.legend()
plt.show()

# Wave path : square wave
time_end = 60
model.reset()

t_data = np.arange(0,time_end,sample_time)
x_data = np.zeros_like(t_data)
y_data = np.zeros_like(t_data)

# maintain velocity at 4 m/s
v_data = np.zeros_like(t_data)
v_data[:] = 4
w_data[:] = 0
w_data[0:100] = 1
w_data[100:300] = -1
w_data[300:500] = 1
w_data[500:5700] = np.tile(w_data[100:500], 13)
w_data[5700:] = -1

for i in range(t_data.shape[0]):
    x_data[i] = model.xc
    y_data[i] = model.yc
    model.step(v_data[i],w_data[i])

plt.axis('equal')
plt.plot(x_data , y_data , label = 'Model trajectory')
plt.legend()
plt.show()

# eight trajectory    

time_end = 30
model.reset()
t_data = np.arange(0,time_end,sample_time)
x_data = np.zeros_like(t_data)
y_data = np.zeros_like(t_data)
v_data = np.zeros_like(t_data)
w_data = np.zeros_like(t_data)

v_data[:] = (16 * np.pi)/15

for i in range(t_data.shape[0]):
    x_data[i] = model.xc
    y_data[i] = model.yc
    
    if i < 352 or i > 1800:
        if model.delta < np.arctan(2/8):
            model.step(v_data[i] , model.w_max)
            w_data[i] = model.w_max
        else:
            model.step(v_data[i] , 0)
            w_data[i] = 0
    else:
        if model.delta > -np.arctan(2/8):
            model.step(v_data[i] , -model.w_max)
            w_data[i] = -model.w_max
        else:
            model.step(v_data[i] , 0)
            w_data[i] = 0
     
    model.beta = 0

plt.axis('equal')
plt.plot(x_data, y_data)
plt.show()
