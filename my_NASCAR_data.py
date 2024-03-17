#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:00:44 2024


So far this is not perfect as actually we should be creating stochastic 
trajectories from a stream plot of the composed LDSs.

Come up with the LDS for each section and then start from randomised initial
conditions.


@author: pzc
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy.integrate import odeint
from scipy.integrate import solve_ivp


def create_NASCAR_data(length, radius, resolution):

    # Generate the top straight sections
    y_top_straight = np.linspace(-length/2, length/2, resolution)
    x_top_straight = np.full_like(y_top_straight, radius)

    # Generate the bottom straight sections
    y_bottom_straight = np.linspace(-length/2, length/2, resolution)
    x_bottom_straight = np.full_like(y_top_straight, -radius)

    # Generate the semicircular ends
    theta_top = np.linspace(np.pi, 2 * np.pi, resolution)
    x_top_semi =  radius * np.cos(theta_top)
    y_top_semi =  radius * np.sin(theta_top) - length/2

    theta_bottom = np.linspace(0, np.pi, resolution)
    x_bottom_semi = radius * np.cos(theta_bottom)
    y_bottom_semi = radius * np.sin(theta_bottom) + length/2

    # Concatenate the arrays to form the racetrack
    
    
    # x_racetrack = np.concatenate((x_top_straight, x_top_semi, x_bottom_straight, x_bottom_semi))
    # y_racetrack = np.concatenate((y_top_straight, y_top_semi, y_bottom_straight, y_bottom_semi))

    # Plot the racetrack
    plt.figure(figsize=(8, 4))
    
    plt.scatter(x_top_straight, y_top_straight, color='orange')
    plt.scatter(x_top_semi, y_top_semi, color='blue')
    plt.scatter(x_bottom_straight, y_bottom_straight, color='green')
    plt.scatter(x_bottom_semi, y_bottom_semi, color='red')

    # plt.scatter(x_racetrack, y_racetrack)
    plt.axis('equal')
    plt.title('NASCAR data')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    
    x_racetrack = np.concatenate((x_top_straight, x_top_semi, x_bottom_straight, x_bottom_semi))
    y_racetrack = np.concatenate((y_top_straight, y_top_semi, y_bottom_straight, y_bottom_semi))
    
    return x_racetrack, y_racetrack


# # Parameters for the racetrack
# length = 10  # Length of the straight sections
# radius = 5   # Radius of the semicirc= ular ends
# resolution = 100  # Number of points per section

# x1, x2 = create_NASCAR_data(length, radius, resolution)


# In[]

def plot_streamplot_single(A, rotation):
    '''
    Do I just plot a streamplot of each separate LDS and then concatenate?
    Do I first carve up the space with linear boundaries?
    First I will try whole space for each plot

    '''
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)  # Adjust figsize and dpi as needed
    figtitle=None
    
    # sample 20 x 20 grid uniformly to get x1 and x2
    grid = np.arange(-20, 21, 1)
    x1, x2 = np.meshgrid(grid, grid)
    
    if rotation=='upper': 
    
        # calculate x1dot and x2dot at each grid point
        x1dot = A[0,0] * x1 + A[0,1] * (x2-10) # offset centre
        x2dot = A[1,0] * x1 + A[1,1] * x2 
        
    elif rotation =='lower':

        # calculate x1dot and x2dot at each grid point
        x1dot = A[0,0] * x1 + A[0,1] * (x2+10) # offset centre
        x2dot = A[1,0] * x1 + A[1,1] * x2
        
    else:
        
        # calculate x1dot and x2dot at each grid point
        x1dot = A[0,0] * x1 + A[0,1] * x2 
        x2dot = A[1,0] * x1 + A[1,1] * x2
        

    # make a colormap
    magnitude = np.sqrt(x1dot ** 2 + x2dot ** 2)
    color = 2 * np.log1p(magnitude) #Avoid taking log of zero

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

    # get eigenvectors of A
    eigenvector1 = v[:,0].real
    eigenvector2 = v[:,1].real

    # plot eigenvectors
    # plt.arrow(0, 0, 20*eigenvector1[0], 20*eigenvector1[1],
    #           width=0.5, color='r', head_width=2,
    #           length_includes_head=True)
    # plt.arrow(0, 0, 20*eigenvector2[0], 20*eigenvector2[1],
    #           width=0.5, color='b', head_width=2,
    #           length_includes_head=True)
    plt.show()


def plot_streamplot():
    '''
    Do I just plot a streamplot of each separate LDS and then concatenate?
    Do I first carve up the space with linear boundaries?

    '''
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)  # Adjust figsize and dpi as needed
    figtitle=None
    
    # sample 20 x 20 grid uniformly to get x1 and x2
    grid = np.arange(-30, 31, 1)
    x1, x2 = np.meshgrid(grid, grid)

    # straight sections: x2 in between -10 and 10
    A1 = np.array([[0,0],[1,0]]) # straight sections 
    # first I create a mask for all elements greater than -10
    mask = (x2 >= -10) & (x2 <= 10)
    x2_masked = np.where(mask, x2, np.nan)
    # calculate x1dot and x2dot at each grid point
    x1dot = A1[0,0] * x1 + A1[0,1] * x2_masked 
    x2dot = A1[1,0] * x1 + A1[1,1] * x2_masked
    
    # upper section: x2 < -10
    A2 = np.array([[0,-10],[10,0]]) # A2 rotation (anti-clockwise)
    rotation = 'upper'
    # first I create a mask for all elements greater than -10
    mask = x2 >= 10
    x2_masked = np.where(mask, x2, np.nan)
    # calculate x1dot and x2dot at each grid point
    x1dotu = A2[0,0] * x1 + A2[0,1] * x2_masked+100 # offset centre (why 100 and not 10?!)
    x2dotu = A2[1,0] * x1 + A2[1,1] * x2_masked
    
    # lower section: x2 > 10:
    A3 = np.array([[0,-10],[10,0]]) # A2 rotation (anti-clockwise)
    rotation = 'lower'
    # first I create a mask for all elements greater than -10
    mask = x2 <= -10
    x2_masked = np.where(mask, x2, np.nan)
    # calculate x1dot and x2dot at each grid point
    x1dotl = A3[0,0] * x1 + A3[0,1] * x2_masked-100 # offset centre
    x2dotl = A3[1,0] * x1 + A3[1,1] * x2_masked
        
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


# A1 = np.array([[0,0],[1,0]]) # straight sections
# A2 = np.array([[0,-10],[10,0]]) # rotation (anti-clockwise)
# plot_streamplot()


'''
# plot_streamplot_single(A1, rotation='none')

# now I need to concatenate them
# essentially say, if x1, x2 between certain boundaries, then give different 
# A matrix dynamics...
# then plot the stream plot...
'''



'''
# need to resolve what is meant by the offset and why it is ok 
# to do what I have done (by adding a bias term to x2 in each rotation case)
# note that I could create a 4 state LDS in which the straight
# sections are governed by two different LDSs
# now I need to sample trajectories

'''

# In[]

def continuous_dynamics(t,x):
    
    #print(x[0], x[1])
    
    noise = np.random.normal(loc=0, scale=0.5, size=2)
    
    A1 = np.array([[0,0],[1,0]]) # straight sections
    A2 = np.array([[0,-10],[10,0]]) # rotation (anti-clockwise)
    
    b1 = np.array([0,0])
    b2 = np.array([-100,0])
    b3 = np.array([100,0])
    
    if x[1] >= -10 and x[1] <= 10:  # straight
        
        A = A1
        b = b1

    
    elif x[1] < -10: # lower
        
        A = A2
        b = b2
        
    elif x[1] > 10: # upper
        
        A = A2
        b = b3
        
    else:
        print('WHOOPS!')
    
    dxdt = np.dot(A, x) + b.T + noise.T # need to make sure bias only for x1dot
    
    return dxdt    

def sample_trajectories(T, num_points, initial_condition):
    
    # select initial_conditions
    x0 = initial_condition
    
    # numerical integration from init conditions to produce trajectory

    # using ODEINT:
    # set time horizon
    # t = np.linspace(0, T, num_points) 
    #x_t = odeint(continuous_dynamics, x0, t)
    
    # using SOLVE_IVP:
    solution = solve_ivp(continuous_dynamics, [0, T], x0, t_eval=np.linspace(0, T, num_points))
    
    # visualise trajectory
    x1 = solution.y[0]
    x2 = solution.y[1]        
    
    plt.plot(x1,x2)
    lim = np.min(x2)-1
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    plt.title(f'Sampled Trajectories from NASCAR Dynamics, T = {T}')
    plt.xlabel(r'$x1$')
    plt.ylabel(r'$x2$')
    plt.show()
    
    return solution
    

def sample_trajectories_colour(T, num_points, initial_condition):
    
    # select initial_conditions
    x0 = initial_condition
    
    # numerical integration from init conditions to produce trajectory

    # using ODEINT:
    # set time horizon
    # t = np.linspace(0, T, num_points) 
    #x_t = odeint(continuous_dynamics, x0, t)
    
    # using SOLVE_IVP:
    solution = solve_ivp(continuous_dynamics, [0, T], x0, t_eval=np.linspace(0, T, num_points))
    
    # visualise trajectory
    x1 = solution.y[0]
    x2 = solution.y[1]   
        
    # Plot the trajectory 
    # plt.scatter(x1,x2, color='black')
    plt.plot(x1,x2, color='black')


    plt.margins(0) # remove default margins (matplotlib verision 2+)

    plt.axhspan(-10, -100, facecolor='cornflowerblue', alpha=0.5)
    plt.axhspan(-10, 10, facecolor='lightgreen', alpha=0.5)
    # plt.axvspan(0, -100, ymin = np.max(x2)/100, ymax = 1-(np.max(x2)/100), facecolor='lightgreen', alpha=0.5)
    # plt.axvspan(0, 100, ymin = np.max(x2)/100, ymax = 1-(np.max(x2)/100), facecolor='gold', alpha=0.5)
    plt.axhspan(10, 100, facecolor='tomato', alpha=0.5)
    
    lim = np.min(x2)-1
    plt.xlim(lim,-lim)
    plt.ylim(lim,-lim)
    plt.title(f'Sampled Trajectories from NASCAR Dynamics, T = {T}')
    plt.xlabel(r'$x1$')
    plt.ylabel(r'$x2$')
    plt.show()
    
    return solution


T = 100
num_points = 1000
initial_condition = [5,5]
sample_trajectories_colour(T, num_points, initial_condition)

'''
why is the bias +100 not +10?!
'''
    




