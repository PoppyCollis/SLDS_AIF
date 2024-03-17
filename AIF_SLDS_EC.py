#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:30:47 2024

Switching linear dynamical system (SLDS) using an Active Inference controller for
Equilibrium Control (EC)

@author: pzc
"""

from pymdp import utils
from agent import Agent
from control import *
import ssm
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import multivariate_normal
from scipy.stats import norm
import continuous_active_inference as caif
from visualization import plot_system_control


# In[] Construct linear dynamics

def plot_streamplot_single(A, b):
    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)  # adjust figsize and dpi as needed
    figtitle=None
    
    # sample 20 x 20 grid uniformly to get x1 and x2
    grid = np.arange(-20, 21, 1)
    x1, x2 = np.meshgrid(grid, grid)
    

    # calculate x1dot and x2dot at each grid point
    x1dot = A[0,0] * x1 + A[0,1] * x2 + b[0]
    x2dot = A[1,0] * x1 + A[1,1] * x2 + b[1]
        

    # make a colormap
    magnitude = np.sqrt(x1dot ** 2 + x2dot ** 2)
    color = 2 * np.log1p(magnitude) # avoid taking log of zero

    # plot
    plt.sca(ax)
    plt.streamplot(x1, x2, x1dot, x2dot, color=color,
                   linewidth=1, cmap=plt.cm.cividis,
                   density=2, arrowstyle='->', arrowsize=1.5)
    plt.xlabel(r'$x1$')
    plt.ylabel(r'$x2$')

    # figure title
    if figtitle is not None:
        plt.title(figtitle, size=16)

    # include eigenvectors
    if True:
        # get eigenvalues and eigenvectors of A
        lam, v = np.linalg.eig(A)

    plt.show()  
    

def create_A_decay(state_dim):
    '''
    Create a random A matrix with negative eigenvalues for decay dynamics
    
    '''
    # select random negative eigenvalues
    #eigen1, eigen2 = np.random.randint(1,5,2)*-1
    eigen1, eigen2 = -3,-3
    
    # create diagonal matrix with eigenvalues
    D = np.array([[eigen1,0],[0,eigen2]])
    
    # select an invertible matrix
    P = np.array([[1,1],[-1,2]])
    
    # check nonzero determinant
    assert np.linalg.det(P) != 0, "P must be an invertible matrix"
    
    A = np.matmul(np.linalg.inv(P), np.matmul(D, P))
    
    return A

def calculate_fp_biases(desired_fp, A):
    '''
    desired fixed point is a vector of dimension state_dim
    '''
    
    b = np.dot(A, -desired_fp)
    
    return b
        
    
state_dim = 2
desired_fp = np.array([5,3])
A_cont = create_A_decay(state_dim)  

fp_0 = np.array([1e-2,1e-2])
fp_1 = np.array([5,10])
fp_2 = np.array([-5,10])
fp_3 = np.array([-5,-10])
fp_4 = np.array([5,-10])

b0 = calculate_fp_biases(fp_0, A_cont)
b1 = calculate_fp_biases(fp_1, A_cont)
b2 = calculate_fp_biases(fp_2, A_cont)
b3 = calculate_fp_biases(fp_3, A_cont)
b4 = calculate_fp_biases(fp_4, A_cont)

fps = [fp_0, fp_1, fp_2, fp_3, fp_4]
bs = [b0,b1, b2, b3, b4]

# for i in range(5):
#     plot_streamplot_single(A_cont, bs[i])
#print(f'fixed point = {np.dot(-np.linalg.inv(A), b)}')

# In[] Construct discrete AIF controller

def build_A_reach(A, fully_known, prob):
    """
    Returns
    -------
    A : A[0] filled of A matrix

    """
    if fully_known:
        
        non_prob=(1-prob)/(A[0].shape[0]-1)
        A[0] = np.ones((A[0].shape[0],A[0].shape[1]))*non_prob
        np.fill_diagonal(A[0], prob)
        
    return A

def fill_A(A, fully_known, prob=1):
    
    build_A_reach(A, fully_known, prob)
    
    return A

def build_B_reach(B, fully_known, prob):
    """
    Note
    -------
    Defines how controls operate on state dimension 0 - reach 
    
    Returns
    -------
    B : fills in B[0]
    """
    if fully_known:
        non_prob = 1-prob
        # action 0 - moves to posn 0:
        B[0][:,:,0] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob / (B[0][:,:,0].shape[0]-1)
        B[0][0,:,0] = prob
        
        # action 0 - moves to posn 1:
        B[0][:,:,1] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob / (B[0][:,:,1].shape[0]-1)
        B[0][1,:,1] = prob
        
        # action 0 - moves to posn 2:
        B[0][:,:,2] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob / (B[0][:,:,2].shape[0]-1)
        B[0][2,:,2] = prob
                
        # action 0 - moves to posn 3:
        B[0][:,:,3] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob / (B[0][:,:,3].shape[0]-1)
        B[0][3,:,3] = prob
        
        # action 0 - moves to posn 4:
        B[0][:,:,4] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob / (B[0][:,:,4].shape[0]-1)
        B[0][4,:,4] = prob           
        
    else:
        # all actions are totally unknown 
        # i.e. B[:,:,: for B[0] - each factor list factor is the same
        for i in range(B[0].shape[2]):
              B[0][:,:,i] = np.ones((B[0].shape[0],B[0].shape[1])) / B[0][:,:,i].shape[0]   
                
    return B

def fill_B(B, fully_known = False, prob = 1):
     
    B = build_B_reach(B, fully_known, prob)
     
    return B

def fill_C(C, punish, reward, reward_location,tiled=False):
    """
    Parameters
    ----------
    C : pymdp obj_array_uniform 
        uniform prior preference array (C vector)
    punish : integer
        relative log probability of observing 0 in reward modality
    reward : integer
        relative log probability of observing 1 in reward modality
    tiled: boolean
        if true then we have a different C across each timestep. 
        otherwise, same C is used across all timesteps
    
    Returns
    -------
    C : C vector
        Filled prior preference over observation array 

    """
    if tiled:
        
       C[0] = np.tile(C[0],(len(reward_location),1))
       C[0][:] = punish
       
       for i in range(len(reward_location)):
           C[0][i,reward_location[i]] = reward
           
       C[0] = C[0].T
    
    else:
        
        C[0][:] = punish
        C[0][reward_location] = reward
        
    return None


# observations
obs_reach = ["Center", "Top-right","Top-left", "Bottom-left", "Bottom-right"]
# states
state_reach = ["Center", "Top-right","Top-left", "Bottom-left", "Bottom-right"]
# controls
reach_action= ["Go-C", "Go-TR","Go-TL", "Go-BL","Go-BR"]
# observation modality dimensions
num_obs = [len(obs_reach)] 
# hidden state factor dimensions
num_states = [len(state_reach)] 
# control state factor dimensions
num_controls = [len(reach_action)] 

# state transitions only depend on own state, so no need for factor lists
B_factor_list=None


# create observation likelihood (A matrix)
A = utils.initialize_empty_A(num_obs, num_states) 
prob_A = 1. # confidence in A matrix
fill_A(A, fully_known = True, prob=prob_A)
dir_scale = 1.
pA=utils.dirichlet_like(A,scale=dir_scale)

# create transition likelihood (B matrix) with factor_list
B = utils.random_B_matrix(num_states, num_controls, B_factor_list=B_factor_list) 
prob_B = 1. # confidence in B matrix
fill_B(B, fully_known = False, prob=prob_B)
dir_scale = 1.
pB = utils.dirichlet_like(B,scale=dir_scale)

# create C matrix and init agent
reward_location = 2
'''
the agent will use reward_location for planning.
also, if reward_location is a list then reward_locaation will be looped. so if is e.g. [4,1,0], 
then at subsequent timestesp will change to [1,0,4],then [0,4,1], then back to [4,1,0]
'''
if isinstance(reward_location,list):
    tiled=True
elif isinstance(reward_location,int):
    tiled=False
else:
    print("reward_location must be an integer or a list")
    
# create prior preference over observations (C vector)
punish, reward = 0, 20
C = utils.obj_array_uniform(num_obs)
if isinstance(reward_location,list): # this is hard coded for 1 observation modality. need to reconsider if more than 1 modality
    for i in range(len(reward_location)):
        assert reward_location[i]<num_obs[0], "reward location cannot be greater than number of discrete observations"
fill_C(C, punish, reward,reward_location,tiled=tiled)

# instantiate agent
policy_len = 3
if isinstance(reward_location,list):
    assert len(reward_location)>=policy_len, "policy length must be the same as len(reward_loation): need a reward location at each time step"
policies = None
policies_restriction= None
my_agent = Agent(A=A, pA=pA, B = B, pB=pB, C = C, policy_len=policy_len, policies_restriction=policies_restriction, policies=policies,B_factor_list=B_factor_list, action_selection="stochastic")

init_position = np.random.randint(len(obs_reach))


run_length=500

plot_charts=False



# In[] Construct SLDS

def continuous_dynamics(t, x, A, b):
    
    noise = np.random.normal(loc=0, scale=1, size=2)
    
    dxdt = np.dot(A, x) + b.T + noise.T
    
    return dxdt  

def run_continuous(T, num_steps, initial_condition, A, b, i):
    
    # select initial_conditions
    x0 = initial_condition
    
    # numerical integration from init conditions to produce trajectory
    # using SOLVE_IVP:
    solution = solve_ivp(continuous_dynamics, [0, T], x0, args=(A, b), t_eval=np.linspace(0, T, num_steps))
    
    # visualise trajectory
    x1 = solution.y[0]
    x2 = solution.y[1]    
    
    plt.plot(x1,x2)
    plt.xlim(-20,20)
    plt.ylim(-20,20)
    plt.title(f'Continuous trajectory step {i}')
    plt.xlabel(r'$x1$')
    plt.ylabel(r'$x2$')
    plt.show()
    
    return x1, x2

# T = 10
# num_steps = 1000
# init_cond = np.array([0,0])
# x1, x2 = run_continuous(T, num_steps, init_cond, A_cont, b1)
# x1s = x1
# x2s = x2

# # init_cond should be the last state of the last run
# x_init = np.array([x1[-1], x2[-1]])
# x1, x2 = run_continuous(T, num_steps, x_init, A_cont, b2)
# x1s = np.concatenate((x1s, x1))
# x2s = np.concatenate((x2s, x2))

# # init_cond should be the last state of the last run
# x_init = np.array([x1[-1], x2[-1]])
# x1, x2 = run_continuous(T, num_steps, x_init, A_cont, b3)
# x1s = np.concatenate((x1s, x1))
# x2s = np.concatenate((x2s, x2))

# # visualise full trajectory
# plt.plot(x1s, x2s)
# plt.xlim(-20,20)
# plt.ylim(-20,20)
# plt.title(f'Full Trajectory')
# plt.xlabel(r'$x1$')
# plt.ylabel(r'$x2$')
# plt.show()
 
# I am expecting to input num time steps, initial condition, A and b and get back trajectory in x1, x2 space
# maybe in this instance I am effectively just plugging in fixed points (not even b)?


def run_continuous_agent(eta, prior_P_f, continuous_agent, system):
    # Receives the desired expected observation tau (desired goal of the system), the continuous agent and the system definition (dynamics, etc)

    # expected_obs is an array of arrays - one array for each modality. 
    # each of those arrays is a discrete probability funciton with a value for each of the psossible discrete outcomes for the modality. 
  
    # # just produces random L_m, for one modality
    # posterior_obs=np.random.rand(expected_obs[0].shape[0])

    # Unrol continous
    F_action = 0
    xd = np.array([[eta], [0]]) # Desired observation [position, velocity]
    print('Prior_P_f', prior_P_f)
    # continuous_agent.set_desired_precision(np.diag([prior_P_f[0][0], prior_P_f[0][0]])) # TODO prior_P_f should be 2-dim diag as it has position and velocity. Only encoding position.
    # continuous_agent.set_desired_precision(np.diag([.01, .01])) # Original value

    for i in range(0, system.nt - 1):
        u, F, Pi_u = continuous_agent.run_aif_step(i, xd, system) # runs one step of VFE minimization (gradient)
        system.update_state(i, u) # Generative process - take action in the world
        F_action += F # Path Integral Free Energy
    # End unrol continous

    x0, y0, u0 = system.get_current_state() # provides current observation

    # Return observations in the real world
    o_prime = system.x.copy()

    plot_system_control('position', system, xd, system.dt, system.nt)
    system.clear_buffer() # TODO discuss how to initalize smoothly

    return o_prime[0], o_prime[1]


# Instantiate Contiuous Agent
# params
nt = 2000
dt = .01 # 0.01
# System
system = caif.SMD(nt=nt, dt=dt) # Define type of system dynamics
# Continuous active inference controller
my_cont_agent = caif.ContiniousAIF(nx=system.nx, ny=system.ny, nu=system.nu)

# In[] Link functions

def calculate_responsibilities(data_point, means, covariance):
    """
    Infer the categorical probability distribution of a given data point 
    belonging to each component of a Gaussian mixture model.

    Parameters:
    - data_point: A 2D continuous data point (numpy array).
    - means: A list of means (numpy arrays) for each Gaussian in the mixture.
    - covariance: A shared diagonal covariance matrix (numpy array).

    Returns:
    - A numpy array representing the categorical probability distribution.
    """
    # Compute the unnormalized responsibilities (likelihoods * priors)
    # Assuming equal priors since not specified
    responsibilities = [multivariate_normal.pdf(data_point, mean=mean, cov=covariance) for mean in means]
    
    # Normalize the responsibilities to get probabilities
    total = sum(responsibilities)
    probabilities = np.array([resp / total for resp in responsibilities])
    
    return probabilities

def calculate_responsibilities_1d(data_point, means, variances):
    """
    Infer the categorical probability distribution of a given data point 
    belonging to each component of a Gaussian mixture model.

    Parameters:
    - data_point: A 2D continuous data point (numpy array).
    - means: A list of means (numpy arrays) for each Gaussian in the mixture.
    - covariance: A shared diagonal covariance matrix (numpy array).

    Returns:
    - A numpy array representing the categorical probability distribution.
    """
    # Compute the unnormalized responsibilities (likelihoods * priors)
    # Assuming equal priors since not specified
    responsibilities = [norm.pdf(data_point, loc=mean, scale=np.sqrt(variance))
                        for mean, variance in zip(means, variances)]
    # Normalize the responsibilities to get probabilities
    total = sum(responsibilities)
    probabilities = np.array([resp / total for resp in responsibilities])
    
    return probabilities


# In[] Plug into AIF agent

def run_active_inference_step(num_runs, my_agent, init_state, T, num_steps, A_cont, bs, fps, fictive_step=False):
    
    x1s = np.zeros((num_runs, num_steps))
    x2s = np.zeros((num_runs, num_steps))
    
    # reward_locations = np.random.randint(0,5,num_runs)
    reward_locations = np.array([1,2,3,4,1,2,3,4])
    print('reward locations: ', reward_locations)
    
    for i in range(num_runs):
    
        reward_location = reward_locations[i]
        # create prior preference over observations (C vector)
        punish, reward = 0, 20
        C = utils.obj_array_uniform(num_obs)
        C[0][:] = punish
        C[0][reward_location] = reward
        
        my_agent.C = C
    
        my_agent.reset() # resets qs and q_pi to uniform
        # plug in end state of last AIF step directly into qs
        next_observation = [np.argmax(init_state)]
        qs = my_agent.infer_states(next_observation)
        print(my_agent.qs)
        
        # infer policies
        q_pi, efe = my_agent.infer_policies_factorized()
        chosen_action_id = my_agent.sample_action()
        
        # find which b is indexed by the action
        b = bs[int(chosen_action_id[0][0])]
        
        # given initial state, init continuous dynamics from relevant coordinates
        x_init = fps[np.argmax(init_state)]
        # step the continuous dynamics
        x1, x2 = run_continuous(T, num_steps, x_init, A_cont, b, i)
        
        x1s[i] = x1
        x2s[i] = x2
        
        if fictive_step:
            # from continuous coordinate, infer posterior responsibilities (E-step)
            sigma = np.array([[1, 0], [0, 1]]) # shared diagonal covariance 
            responsibilities = calculate_responsibilities(np.array([x1[-1], x2[-1] ]), fps, sigma)
            state = np.argmax(responsibilities)
            init_state = np.eye(len(fps))[state]
    
        else:
            # just get next state by applying transition matrix and action
            transition = B[0][...,int(chosen_action_id[0][0])]
            init_state = np.dot(transition, init_state)
        
    for i in range(num_runs):
        plt.plot(x1s[i], x2s[i])
    plt.xlim(-20,20)
    plt.ylim(-20,20)
    plt.title(f'Full Trajectory')
    plt.xlabel(r'$x1$')
    plt.ylabel(r'$x2$')
    plt.show()
    return x1s, x2s, init_state

def run_active_inference_step_with_learning(num_runs, my_agent, init_state, T, num_steps, A_cont, bs, fps, fictive_step = False):
    
    x1s = np.zeros((num_runs, num_steps))
    x2s = np.zeros((num_runs, num_steps))
    
    #reward_locations = np.random.randint(0,5,num_runs)
    reward_locations = np.array([1,2,3,4])
    print('reward locations: ', reward_locations)
    
    for i in range(num_runs):
    
        reward_location = reward_locations[i]
        # create prior preference over observations (C vector)
        punish, reward = 0, 20
        C = utils.obj_array_uniform(num_obs)
        C[0][:] = punish
        C[0][reward_location] = reward
        
        my_agent.C = C
    
        my_agent.reset() # resets qs and q_pi to uniform
        # plug in end state of last AIF step directly into qs
        next_observation = [np.argmax(init_state)]
        qs = my_agent.infer_states(next_observation)
        print(my_agent.qs)
        
        # update A and B matrices 
        if i > 0: # the action carried out to initilise shouldn't be used for learning
            qA = my_agent.update_A(next_observation)   
            qB = my_agent.update_B(qs_prev)
        
        qs_prev = qs
        
        # infer policies
        q_pi, efe = my_agent.infer_policies_factorized()
        chosen_action_id = my_agent.sample_action()
        
        # find which b is indexed by the action
        b = bs[int(chosen_action_id[0][0])]
        
        # given initial state, init continuous dynamics from relevant coordinates
        x_init = fps[np.argmax(init_state)]
        # step the continuous dynamics
        x1, x2 = run_continuous(T, num_steps, x_init, A_cont, b, i)
        
        x1s[i] = x1
        x2s[i] = x2
        
        if fictive_step:
            # from continuous coordinate, infer posterior responsibilities (E-step)
            sigma = np.array([[1, 0], [0, 1]]) # shared diagonal covariance 
            responsibilities = calculate_responsibilities(np.array([x1[-1], x2[-1] ]), fps, sigma)
            state = np.argmax(responsibilities)
            init_state = np.eye(len(fps))[state]
    
        else:
            # just get next state by applying transition matrix and action
            transition = B[0][...,int(chosen_action_id[0][0])]
            init_state = np.dot(transition, init_state)
        
    for i in range(num_runs):
        plt.plot(x1s[i], x2s[i])
    plt.xlim(-20,20)
    plt.ylim(-20,20)
    plt.title(f'Full Trajectory')
    plt.xlabel(r'$x1$')
    plt.ylabel(r'$x2$')
    plt.show()
    
    print(my_agent.B)
    
    
    return x1s, x2s, init_state


def run_active_inference_step_CAIF(num_runs, my_agent, init_state, fps, nt, dt, system, my_cont_agent, fictive_step=True):
    
    x1s = np.zeros((num_runs, nt))
    x2s = np.zeros((num_runs, nt))
    
    # note for CAIF, position in just 1 dimensional
    fps = [0,1,2,3,4] # 5 positions on a 1D plane so fps are same as self.etas
    
    reward_locations = np.random.randint(0,5,num_runs)
    # reward_locations = np.array([1,2,3,4])
    print('reward locations: ', reward_locations)
    
    for i in range(num_runs):
        
        # create prior preference over observations (C vector)
        reward_location = reward_locations[i]
        punish, reward = 0, 20
        C = utils.obj_array_uniform(num_obs)
        C[0][:] = punish
        C[0][reward_location] = reward
        my_agent.C = C
    
        my_agent.reset() # resets qs and q_pi to uniform
        # plug in end state of last AIF step directly into qs
        next_observation = [np.argmax(init_state)]
        qs = my_agent.infer_states(next_observation)
        print(my_agent.qs)
        
        # update A and B matrices 
        if i > 0: # the action carried out to initilise shouldn't be used for learning
            qA = my_agent.update_A(next_observation)   
            qB = my_agent.update_B(qs_prev)
        
        qs_prev = qs
        
        # infer policies
        q_pi, efe = my_agent.infer_policies_factorized()
        chosen_action_id = my_agent.sample_action()
        
        # find which fp is indexed by the action
        eta = fps[int(chosen_action_id[0][0])]
        
        # given initial state, init continuous dynamics from relevant coordinates
        x_init = fps[np.argmax(init_state)]
        
        prior_P_f= np.diag(np.ones(1) * 0.1)
        
        
        # step the continuous dynamics
        x1, x2 = run_continuous_agent(eta, prior_P_f, my_cont_agent, system)
        
        x1s[i] = x1
        x2s[i] = x2
        
        if fictive_step:
            # from continuous coordinate, infer posterior responsibilities (E-step)
            sigmas = np.ones(len(fps)) # shared diagonal covariance 
            # calculate 1D responsibilities from normal distributions
            responsibilities = calculate_responsibilities_1d(np.array([x1[-1]]), fps, sigmas)
            state = np.argmax(responsibilities)
            init_state = np.eye(len(fps))[state]
    
        else:
            # just get next state by applying transition matrix and action
            transition = B[0][...,int(chosen_action_id[0][0])]
            init_state = np.dot(transition, init_state)
        
    for i in range(num_runs):
        plt.plot(x1s[i], x2s[i])
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.title(f'Full Trajectory')
    plt.xlabel(r'$x1$')
    plt.ylabel(r'$x2$')
    plt.show()
    return x1s, x2s
    
# In[] Run experiment

num_runs = 50
T = 10
num_steps = 100
init_state = np.array([0,1,0,0,0]) # start at top-right

# x1s, x2s, init_state = run_active_inference_step(num_runs, my_agent, init_state, T, num_steps, A_cont, bs, fps, fictive_step=True)
# x1s, x2s, init_state = run_active_inference_step_with_learning(num_runs, my_agent, init_state, T, num_steps, A_cont, bs, fps, fictive_step=True)
x1s, x2s = run_active_inference_step_CAIF(num_runs, my_agent, init_state, fps, nt, dt, system, my_cont_agent, fictive_step=True)


    
    




