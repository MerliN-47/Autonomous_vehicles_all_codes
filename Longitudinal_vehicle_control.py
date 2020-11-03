# Forward lonigtudinal vehicle model which accepts throttle inputs and steps through 
# the longitudinal dynamic equations. 
# Input is throttle percentage x_theta in range [0,1] providing torque to the engine and also accelerating the vehicle forward 
# Dynamic equations consist of many stages to convert throttle inputs to wheel speed
# ENGINE -> TORQUE CONVERTER ->  TRANSMISSION -> WHEEL 
# Single inertial term J_e in the following combined engine dynamic equations

# J_e * w_e_dot  = T_e - (GR) * (r_eff * F_load)
# m * x_dot_dot = F_x - F_load
# T_e -> engine torque
# GR -> Gear ratio 
# r_eff -> effective radius 
# F_load -> total load force
# m -> vehicle mass 
# x -> vehicle postion
# F_x -> tire force
# F_load -> total load force

# Engine torque is calculaed from the throttle input and engine angular velocity w_e using 
# quadratic model 
# T_e = x_theta * (a_o + a_1*w_e + a_2*(w_e)^2)
# Load forces consist of aerodynamic drag F_aero , rolling friction R_x and the gravitational force F_g from an incline angle alpha.
# Aerodynamic drag is quadratic model and friction is linear model

# F_load = F_aero + R_x + F_g
# F_load = (1/2)* C_a * rho * A * (x_dot)^2 = c_a * (x_dot)^2 
# F_g = m*g*sin(alpha)

# Tire force is computed using engine speed and wheel slip eqns
# w_omega = (GR) * w_e
# s = (w_omega * r_e - x_dot) / x_dot
# F_x = c*s if abs(s) < 1 else F_max 
# w_omega is wheel angular velocity and s is the slip ratio
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Vechicle():
    def __init__(self):
        #Throttle to engine torque
        self.a_0 = 400
        self.a_1 = 0.1
        self.a_2 = -0.0002
        #Gear ratio , effective radius , mass + inertia
        self.GR = 0.35
        self.r_e = 0.3
        self.J_e = 10
        self.m = 2000
        self.g = 9.81
        #Aerodynamic and friction coefficients
        self.c_a = 1.36
        self.c_r1 = 0.01
        #Tire force
        self.c = 10000
        self.F_max = 10000
        #State variables 
        self.x = 0
        self.v = 5
        self.a = 0
        self.w_e = 100
        self.w_e_dot = 0
        
        self.sample_time = 0.01
        
    def reset(self):
        self.x = 0
        self.v = 0
        self.a = 0
        self.w_e = 0
        self.w_e_dot = 0

# Function step takes throttle x_theta and incline angle alpha as iput and performs numerical integration over timestep 
# to update state variables 
class Vehicle(Vehicle):
    def step (self , throttle , alpha):
        F_aero = self.c_a * self.v * self.v
        R_x = self.c_r1 * self.v
        F_g = self.m * self.g * np.sin(alpha)
        F_load = F_aero + R_x + F_g
        
        T_e = throttle * (self.a_0 + self.a_1 * self.w_e + self.a_2 * self.w_e * self.w_e)
        W_w = self.GR * self.w_e 
        r_eff = self.v / W_w
        s = (W_w * self.r_e - self.v) / self.v
        cs = self.c * s
        
        if abs(s) < 1:
            F_x = cs
        else:
            F_x = self.F_max
            
        self.x = self.x + self.v * self.sample_time
        self.v = self.v + self.a * self.sample_time
        self.a = (F_x - F_load) / (self.m)
        self.w_e = self.w_e + self.w_e_dot * self.sample_time 
        self.w_e_dot = (T_e - self.GR * self.r_e * F_load) / self.J_e
     
# Using this we can send throttle inputs to the vehicle, notce how velocity converges to
# fixed value based on throttle input due to aerodynamic drag and tire force limit.
    
sample_time = 0.01
time_end = 100
model = Vehicle()
t_data = np.arange(0 , time_end , sample_time)
v_data = np.zeros_like(t_data)
throttle = 0.1
alpha = 0
for i in range(t_data.shape[0]):
    v_data[i] = model.v
    model.step(throttle , alpha)
plt.plot(t_data , v_data)
plt.show()

# Implementing a model in which vehicle begins at 20% throttle and gradually increases to 50% throttle, maintained for 10 seconds 
# and then vehicle reduces the throttle to 0.
# Implementing ramp angle profile alpha(x) and throttle x_theta(t) and step them through the equations
# Vehicle position x(t) is saved in x_data.

time_end = 20
t_data = np.arange(0 , time_end , sample_time)
x_data = np.zeros_like(t_data)
v_data = np.zeros_like(t_data)
w_e_data = np.zeros_like(t_data)

model.reset()

def angle(i , alpha , x):
    if x < 60:
        alpha[i] = np.arctan(3/60)
    elif x < 150:
        alpha[i] = np.arctan(9/90)
    else:
        alpha[i] = 0

throttle = np.zeros_like(t_data)
alpha = np.zeros_like(t_data)

for i in range(t_data.shape[0]):
    if t_data[i] < 5:
        throttle[i] = 0.2 + ((0.5 - 0.2)/5)*t_data[i]
        angle(i, alpha, model.x)
    elif t_data[i] < 15:
        throttle[i] = 0.5
        angle(i, alpha, model.x)
    else:
        throttle[i] = ((0 - 0.5)/(20 - 15))*(t_data[i] - 20)
        angle(i, alpha, model.x)
    
    #call the step function and update x_data array
    model.step(throttle[i], alpha[i])
    x_data[i] = model.x
    v_data[i] = model.v
    w_e_data[i] = model.w_e
    
plt.title('Distance')
plt.plot(t_data, x_data)
plt.show()

plt.title('Velocity')
plt.plot(t_data, v_data)
plt.show()

plt.title('w_e')
plt.plot(t_data, w_e_data)
plt.show()

plt.title('throttle')
plt.plot(t_data, throttle)
plt.show()

plt.title('alpha')
plt.plot(t_data, alpha)
plt.show()


