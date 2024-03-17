# Continuous time active Inference Controller for a spring mass damper system.
# Author: Ajith Anil Meera, Donders Institute for Brain, Cognition and Behaviour, Nijmegen
# Ported to Python by: Pablo Lanillos, Spanish National Research Council

# Date: 31st October 2023
# Code developed for the Tutorial on Continuous Active Inference at the International Conference on Active Inference IWAI 2023.

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import pinv, expm, det
from scipy.signal import cont2discrete # For transforming the Continuous plant to Discrete
from visualization import SMDVisualizer, plot_system_control # Import SMD visualization class



# general linear system in state space equations
class LTI_model():
    def __init__(self, A, B, C, D, dt):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.dt = dt
        self.sys_d = cont2discrete((A, B, C, D), dt)  # discretized LTI


# spring mass damper system k = .2 m = .4 b = .4
class SMD(LTI_model):
    def __init__(self, k=.2, m=.4, b=.4, nt=2000, dt=.01):
        super(SMD, self).__init__(A=np.array([[0, 1], [-k / m, - b / m]]), B=np.array([[0], [1 / m]]), C=np.array([[1, 0]]), D=np.array([[0]]), dt=dt)
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]
        self.ny = self.C.shape[0]
        self.u = np.zeros((self.nu, nt))  # repmat(1 * sin(.1 * (0:dt:(nt-1) * dt)), nu, 1) # k1 sin(omega * t)
        self.x = np.zeros((self.nx, nt))
        self.y = np.zeros((self.ny, nt))
        self.nt = nt
        self.i = 0 # instant

    # get current state
    def get_current_state(self):
        return self.x[:,self.i], self.y[:,self.i], self.u[:,self.i-1]

    # Update physical state
    def update_state(self, i, u):
        self.i = i
        self.u[:, i] = u
        self.x[:, i + 1] = self.sys_d[0] @ self.x[:, i] + self.sys_d[1] @ self.u[:,i]  # note that we use the discretized LTI
        self.y[:, i] = self.sys_d[2] @ self.x[:, i]  # + .001 * (-1 + 2 * rand(ny, 1))

    def clear_buffer(self):
        self.x[:,0], self.y[:,0], self.u[:,0] = self.get_current_state()
        self.i = 0

    def reset(self):
        self.u = np.zeros((self.nu, self.nt))
        self.x = np.zeros((self.nx, self.nt))
        self.y = np.zeros((self.ny, self.nt))
        self.i = 0



class ContiniousAIF:
    def __init__(self, nx, ny, nu):
        self.x = np.zeros((nx,1)) # state
        self.y = np.zeros((ny,1)) # observation
        self.a = np.zeros((nu, 1))  # action
        self.Pa = .00001 * np.eye(nu)  # prior precision on action # A = eye(nx)
        self.Pi_w = 10 * np.eye(nx)
        self.k_h = 1
        self.Pi_vel = np.diag([1., 1.]) # prior precision on velocity
        self.Pi_g = np.diag([.01, .01])  # goal precision(gain): tune it with dt for stability of Euler discretization
        self.eta = {'Fully Extended':0.0,'Partially Extended':1.0, 'Mid Point':2.0, 'Partially Compressed':3.0,'Fully Compressed':4.0}
        self.k_a = 1
        self.x_d = np.array([[self.eta['Fully Compressed']], [0]]) # initial desired goal

    def reset(self):
        self.x = np.zeros_like(self.x)  # state
        self.y = np.zeros_like(self.y)  # observation
        self.a = np.zeros_like(self.a)  # action

    # maps categorical value into eta continuous desired goal
    def goal2eta(self, cat_model):
        try:
            goal = np.array([[self.eta[cat_model]], [0]])
        except KeyError:
            return np.array([])

        # maps categorical value into eta continuous desired goal
    def obs2eta(self, cat_obs):
        try:
            return np.array([self.eta[x] for x in cat_obs])
        except KeyError:
            return []

    def set_desired_precision(self, precision_desired):
        self.Pi_g = precision_desired


    def set_desired_observation(self, observation_desired):
        self.x_d = observation_desired

    # only for testing purposes
    # def get_goal(self,time):
    #     desired_goal = self.eta['Fully Compressed']
    #     if 5 < time < 10:
    #         desired_goal = self.eta['Partially Extended']
    #     elif 10 <= time < 15:
    #         desired_goal = self.eta['Partially Compressed']
    #     elif time >= 15:
    #         desired_goal = self.eta['Fully Extended']
    #     return np.array([[desired_goal], [0]])

    def run_aif_step(self, i, x_desired, system):
        # assuming a known state
        self.x = np.asmatrix(system.x[:, i]).T  # brain.x = Dx - dFdx Assuming perfect perception
        # Action from FEP (without priors on action):dadt = -dFda = -dFdx * dxda

        # Position control
        # active inference for constant goal state
        # a(:, i) = a(:, i - 1) - learn * ((brain.x - goal_x)'*Pi_g*(-inv(model.A)*model.B) + Pa*a(:,i-1)  )*dt
        dFda = (self.x - x_desired).T @ self.Pi_g @ (-pinv(system.A) @ system.B) + self.Pa @ self.a
        dFdaa = (-pinv(system.A) @ system.B).T @ self.Pi_g @ (-pinv(system.A) @ system.B) + self.Pa

        self.a = self.a + self.k_a * (expm(-self.k_h * dFdaa * system.dt) - np.eye(system.ny)) @ (
                    pinv(dFdaa) @ dFda)  # action update from Free energy curve

        # Only for evaluating VFE
        Pi_u = dFdaa  # precision on action
        Ug = .5 * (self.x - x_desired).T @ self.Pi_g @ (self.x - x_desired)  # performance
        Uc = .5 * self.a.T @ self.Pa @ self.a  # control cost
        H = - .5 * np.log(det(Pi_u))  # confidence
        F = Ug + Uc + H  # free energy

        # print('Ug, Uc, H, F', Ug, Uc, H, F)
        return self.a, F, Pi_u

# # Init params
# task = 'position' # position, velocity, acceleration control
# goal_x = np.array([[.5],[0]]) # goal position and velocity
# dt = .1 # sampling time nt = 2000
# nt = 200 # number of time steps
#
# # initialize the system
# system = SMD(nt=nt, dt=dt)
# brain = ContiniousAIF(nx=system.nx, ny=system.ny, nu=system.nu)
# x0,u0 = system.get_current_state()
# xd = np.zeros((system.nx, nt))  # for debuggin goal generation
#
#
# # VFE and control confidence
# Pi_u = 0 # precision on action
# F_action = 0 # VFE time integral of free energy
#
# for i in range(1, nt - 1):
#     goal_x = brain.get_goal(i * dt) # define goal:
#     xd[:, i] = goal_x.flatten()
#     u, F, Pi_u = brain.run_aif_step(i, goal_x, system)
#     # Generative process - take action in the world
#     system.update_state(i, u)
#     F_action += F
#     print('F', F)


# system.u[:, -1] = system.u[:, -2]
# # Display results
# # plot_results('position', system, xd, dt, nt)
#
#
# # Animation
# # System update callback Function
# def update_visualization(arg):
#     return system.x[0, arg], system.x[1, arg], system.u[0, arg], xd[0,arg], xd[1,arg]  # Return position
#
# smdvisual = SMDVisualizer(callback=update_visualization, interval=dt*1000, simulation_time=dt*nt-1, \
#                           initial=(x0[0], goal_x[0], x0[1], goal_x[1], u0))
#
#
# print('end')
