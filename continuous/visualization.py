# SMD visualization
# Adapted from @senceryazici coding example https://github.com/enesdemirag/programming-exercises/blob/master/exercises/materials/mass-spring-damper-simulation/visualizer.py

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


class SMDVisualizer(object):
    def __init__(self, callback=None, interval=100.0, simulation_time=20.0, initial=(0, 0, 0, 0, 0)):
        # Set loop interval
        self.interval = interval
        self.msfactor = 1000
        self.simulation_time = simulation_time
        # Create figure
        lim = (-1.0,6.0)
        self.smd_size = (2, 0.4)
        self.fig = plt.figure(1, figsize=(13, 6), dpi=80)
        # Create animation plot area
        self.animation_plot = self.fig.add_subplot(223)
        # Create position plot
        self.position_plot = self.fig.add_subplot(221)
        self.position_plot.set_xlim((0, self.simulation_time))
        self.position_plot.set_ylim((lim[0], lim[1]))
        self.position_plot_patch = GraphPlot()
        self.position_plot.add_patch(self.position_plot_patch.first_patch)
        self.position_plot.add_patch(self.position_plot_patch.second_patch)
        self.position_plot.text(0.85, 0.8, 'Red:  x\nBlue: Xref', transform=self.position_plot.transAxes)
        self.position_plot.text(0.03, 0.9, 'Position Plot', transform=self.position_plot.transAxes)
        # Create velocity plot
        self.velocity_plot = self.fig.add_subplot(222)
        self.velocity_plot.set_xlim((0, self.simulation_time))
        self.velocity_plot.set_ylim((lim[0], lim[1]))
        self.velocity_plot_patch = GraphPlot()
        self.velocity_plot.add_patch(self.velocity_plot_patch.first_patch)
        self.velocity_plot.add_patch(self.velocity_plot_patch.second_patch)
        self.velocity_plot.text(0.83, 0.8, 'Red:  Vx\nBlue: Vref', transform=self.velocity_plot.transAxes)
        self.velocity_plot.text(0.03, 0.9, 'Velocity Plot', transform=self.velocity_plot.transAxes)
        # Create acceleration plot
        self.force_plot = self.fig.add_subplot(224)
        self.force_plot.set_xlim((0, self.simulation_time))
        self.force_plot.set_ylim((lim[0], lim[1]))
        self.force_plot_patch = GraphPlot()
        self.force_plot.add_patch(self.force_plot_patch.first_patch)
        self.force_plot.add_patch(self.force_plot_patch.second_patch)
        self.force_plot.text(0.83, 0.8, 'Red:  u\nBlue: 0', transform=self.force_plot.transAxes)
        self.force_plot.text(0.03, 0.9, 'Force Plot', transform=self.force_plot.transAxes)
        # Create dynamic label for simulation time
        self.sim_time_label = self.animation_plot.text(0.03, 0.9, '', transform=self.animation_plot.transAxes)
        # Set axes limit
        self.animation_plot.set_xlim((lim[0], lim[1]))
        self.animation_plot.set_ylim((0, 2))
        # Create the mass object
        self.patch = plt.Rectangle((0, 0), width=self.smd_size[0], height=self.smd_size[1], fc='b')
        # Add the patches
        self.animation_plot.add_patch(self.patch)
        # Create animation core
        self.animate = animation.FuncAnimation(self.fig, self.animate, interval=self.interval,
                                               blit=True, repeat=True, init_func=self.init_func)
        # Set initials to zero
        self.sim_time = 0
        self.x = initial[0] # position state
        self.y = initial[1] # desired position
        self.vx = initial[2] # velocity state
        self.vy = initial[3] # desired velocity
        self.ux = initial[4] # force applied


        # Set callback
        self.cb_func = callback
        plt.show()

    def init_func(self):
        # Set initials to zero
        self.sim_time_label.set_text("")
        self.patch.xy = (0, 0)

        pos_plot = self.position_plot_patch.init_func()
        vel_plot = self.velocity_plot_patch.init_func()
        force_plot = self.force_plot_patch.init_func()
        ret_ = [self.patch, self.sim_time_label] + force_plot + vel_plot + pos_plot
        # return the objects to be redrawn
        return ret_

    def animate(self, i):
        # update simulation time, i = n'th time that the method has
        # called, thus each represents self.interval ms of delay.
        self.sim_time = float(i) / (self.msfactor / self.interval)

        if self.sim_time > self.simulation_time:
            self.animate.event_source.stop()

        # Update time
        self.sim_time_label.set_text("Simulation Time: " + str(self.sim_time) + " / " + str(self.simulation_time))

        # Get object xy from callback
        x, vx, u, zx, zv = self.cb_func(i)

        # Set the position of object
        self.x = x
        self.vx = x
        self.ux = u
        self.y = zx
        self.vy = zv

        # vx = (x - self.x) / (self.interval / 1000.0)
        # self.ax = (vx - self.vx) / (self.interval / 1000.0)

        self.patch.xy = (x - self.smd_size[0]/2.0, 0)

        pos_plot = self.position_plot_patch.update([[self.sim_time, self.x], [self.sim_time, self.y]])
        vel_plot = self.velocity_plot_patch.update([[self.sim_time, self.vx], [self.sim_time, self.vy]])
        force_plot = self.force_plot_patch.update([[self.sim_time, self.ux], [self.sim_time, self.vy]])
        ret_ = [self.patch, self.sim_time_label] + force_plot + vel_plot + pos_plot
        # return the objects to be redrawn
        return ret_


class GraphPlot():
    def __init__(self):
        self.first_patch = plt.Polygon([[0, 0]], closed=None, fill=None, edgecolor='r')
        self.second_patch = plt.Polygon([[0, 0]], closed=None, fill=None, edgecolor='b')

    def init_func(self):
        self.first_patch.xy = [[0, 0]]
        self.second_patch.xy = [[0, 0]]
        return [self.first_patch, self.second_patch]

    def update(self, data):
        # Add current x position to x graph
        arr = self.first_patch.get_xy().tolist()
        arr.append(data[0])
        self.first_patch.xy = arr

        # Add current y position to y graph
        arr = self.second_patch.get_xy().tolist()
        arr.append(data[1])
        self.second_patch.xy = arr

        return [self.first_patch, self.second_patch]



# Stand alone plot controller results
def plot_system_control(task, system, goal, dt, nt):

    print('Controller: ', task) # Type of controller

    fig, ax = plt.subplots(4, 1)

    ax[0].plot(np.arange(0, np.ceil((nt)*dt), dt), system.x[0,:], linewidth=2)
    ax[0].axhline(y=goal[0], linestyle='dashed', color='black', linewidth=2)
    #ax[0].plot(np.arange(0, np.ceil((nt - 1)* dt), dt), goal[0,:], linestyle='dashed', color='black', linewidth=2)
    ax[0].set_xlabel('time (s)')
    ax[0].set_ylabel('x')

    ax[1].plot(np.arange(0, np.ceil((nt - 1)*dt), dt) , system.x[1,:].T,linewidth=2)
    ax[1].set_xlabel('time (s)')
    ax[1].set_ylabel('$\dot{x}$')

    ax[2].plot(np.arange(dt, (nt - 1)*dt, dt), np.diff(system.x[1,:-1]).T/dt, linewidth=2)
    ax[2].set_xlabel('time (s)')
    ax[2].set_ylabel('$\ddot{x}$')


    ax[3].plot(np.arange(0, np.ceil((nt - 1)*dt), dt), system.u.T, linewidth=2)
    ax[3].set_xlabel('time (s)')
    ax[3].set_ylabel('u')
    plt.show()