#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:23:42 2024

Switching linear dynamical system (SLDS) using an Active Inference controller for
Trajectory Control (EC)

@author: pzc
"""

from pymdp import utils
from agent import Agent
import ssm
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


# In[] Construct linear dynamics

    
def plot_streamplot():

    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)  # Adjust figsize and dpi as needed
    figtitle=None
    
    # sample 20 x 20 grid uniformly to get x1 and x2
    grid = np.arange(-30, 31, 1)
    x1, x2 = np.meshgrid(grid, grid)

    # straight sections: x2 in between -10 and 10
    A0 = np.array([[0,0],[10,0]]) # straight sections 
    # first I create a mask for all elements greater than -10
    mask = (x2 >= -10) & (x2 <= 10)
    x2_masked = np.where(mask, x2, np.nan)
    # calculate x1dot and x2dot at each grid point
    x1dot = A0[0,0] * x1 + A0[0,1] * x2_masked 
    x2dot = A0[1,0] * x1 + A0[1,1] * x2_masked
    
    # upper section: x2 < -10
    A1 = np.array([[0,-10],[10,0]]) # rotation upper
    rotation = 'upper'
    # first I create a mask for all elements greater than -10
    mask = x2 >= 10
    x2_masked = np.where(mask, x2, np.nan)
    # calculate x1dot and x2dot at each grid point
    x1dotu = A1[0,0] * x1 + A1[0,1] * x2_masked+100 # offset centre (why 100 and not 10?!)
    x2dotu = A1[1,0] * x1 + A1[1,1] * x2_masked
    
    # lower section: x2 > 10:
    A2 = np.array([[0,-10],[10,0]]) # rotation lower
    rotation = 'lower'
    # first I create a mask for all elements greater than -10
    mask = x2 <= -10
    x2_masked = np.where(mask, x2, np.nan)
    # calculate x1dot and x2dot at each grid point
    x1dotl = A2[0,0] * x1 + A2[0,1] * x2_masked-100 # offset centre
    x2dotl = A2[1,0] * x1 + A2[1,1] * x2_masked
        
    # make a colormap
    magnitude1 = np.sqrt(x1dotu ** 2 + x2dotu ** 2)
    color1 = 2 * np.log1p(magnitude1) #Avoid taking log of zero
    
    magnitude2 = np.sqrt(x1dotl ** 2 + x2dotl ** 2)
    color2 = 2 * np.log1p(magnitude2) #Avoid taking log of zero
    
    magnitude3 = np.sqrt(x1dot ** 2 + x2dot ** 2)
    color3 = 2 * np.log1p(magnitude3) #Avoid taking log of zero

    # plot
    plt.sca(ax)
    plt.streamplot(x1, x2, x1dotu, x2dotu, color=color1,
                    linewidth=1, cmap=plt.cm.cividis,
                    density=1, arrowstyle='->', arrowsize=1.5)
    plt.streamplot(x1, x2, x1dotl, x2dotl, color=color2,
                    linewidth=1, cmap=plt.cm.cividis,
                    density=1, arrowstyle='->', arrowsize=1.5)
    plt.streamplot(x1, x2, x1dot, x2dot, color=color3,
                    linewidth=1, cmap=plt.cm.cividis,
                    density=1, arrowstyle='->', arrowsize=1.5)
    
    
    plt.xlabel(r'$x1$')
    plt.ylabel(r'$x2$')

    # figure title
    if figtitle is not None:
        plt.title(figtitle, size=16)

    # include eigenvectors
    if True:
        # get eigenvalues and eigenvectors of A
        lam, v = np.linalg.eig(A1)

    # # get eigenvectors of A
    # eigenvector1 = v[:,0].real
    # eigenvector2 = v[:,1].real
    #
    # # plot eigenvectors
    # plt.arrow(0, 0, 20*eigenvector1[0], 20*eigenvector1[1],
    #           width=0.5, color='r', head_width=2,
    #           length_includes_head=True)
    # plt.arrow(0, 0, 20*eigenvector2[0], 20*eigenvector2[1],
    #           width=0.5, color='b', head_width=2,
    #           length_includes_head=True)
    
    plt.title('Streamplot of Synthetic NASCAR')
    plt.show()


plot_streamplot()

A0 = np.array([[0,0],[10,0]]) # straight sections
A1 = np.array([[0,-8],[8,0]]) # rotation lower
A2 = np.array([[0,0],[10,0]]) # straight sections
A3 = np.array([[0,-8],[8,0]]) # rotation upper
As = [A0, A1, A2, A3]

b0 = np.array([0,0])
b1 = np.array([100,0])
b2 = np.array([0,0])
b3 = np.array([-100,0])
bs = [b0, b1, b2, b3]

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
obs_reach = ["rotate-upper", "straight-left", "rotate-lower", "straight-right"]
# states
state_reach = ["rotate-upper", "straight-left", "rotate-lower", "straight-right"]
# controls
reach_action= ["Go-RU", "Go-SL","Go-RL", "Go-SR"]
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
fill_A(A, fully_known=True, prob=prob_A)
dir_scale = 1.
pA=utils.dirichlet_like(A,scale=dir_scale)

# create transition likelihood (B matrix) with factor_list
B = utils.random_B_matrix(num_states, num_controls, B_factor_list=B_factor_list) 
prob_B = 1. # confidence in B matrix
fill_B(B, fully_known = True, prob=prob_B)
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


# In[] Construct SLDS

def continuous_dynamics(t, x, A, b):
    
    noise = np.random.normal(loc=0, scale=50, size=2)
    
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
    plt.xlim(-40,40)
    plt.ylim(-40,40)
    plt.title(f'Continuous trajectory step {i}')
    plt.xlabel(r'$x1$')
    plt.ylabel(r'$x2$')
    plt.show()
    
    return x1, x2


# In[] Plug into AIF agent

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

def run_active_inference_step(num_runs, my_agent, init_state, T, num_steps, As, bs, fictive_step=False):
    

    x1s = np.zeros((num_runs, num_steps))
    x2s = np.zeros((num_runs, num_steps))
    
    x_init = np.array([20,20])
    
    # reward_locations = np.random.randint(0,5,num_runs)
    reward_locations = np.array([0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0])
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
        
        # find which A is indexed by the action
        A = As[int(chosen_action_id[0][0])]
        b = bs[int(chosen_action_id[0][0])]
        
        # given initial state, init continuous dynamics from relevant coordinates
        # step the continuous dynamics
        x1, x2 = run_continuous(T, num_steps, x_init, A, b, i)
        x1_prev, x2_prev = x1, x2
        
        x1s[i] = x1
        x2s[i] = x2
        
        x_init = np.array([x1[-1], x2[-1]])
        
    
        
        # just get next state by applying transition matrix and action
        transition = B[0][...,int(chosen_action_id[0][0])]
        init_state = np.dot(transition, init_state)
        
    for i in range(num_runs):
        plt.plot(x1s[i], x2s[i])
    plt.xlim(-100,100)
    plt.ylim(-100,100)
    plt.title(f'Full Trajectory')
    plt.xlabel(r'$x1$')
    plt.ylabel(r'$x2$')
    plt.show()
    return x1s, x2s, init_state

num_runs = 17
T = 0.2
num_steps = 100
init_state = np.array([1,0,0,0,]) # start at top-right

x1s, x2s, init_state = run_active_inference_step(num_runs, my_agent, init_state, T, num_steps, As, bs, fictive_step=False)




