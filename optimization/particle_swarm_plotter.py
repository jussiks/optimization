#!/usr/bin/env python

"""Plotter displays positions of the particles during each iteration
as well as the evolution of the best value.

author: jussiks
"""

import numpy
import matplotlib.pyplot as plt


class Plotter:
    """Plotting object for diplaying charts during swarm optimization
    algorithm.
    """

    def __init__(self, swarm_plot_domain=None, value_plot_domain=None,
                 iterations_to_plot=100, sleep=0.2, func_str=None):
        """Plotter initialization.

        Initializes a scatter plot to display particle positions and a line
        chart to display best value.
        """
        swarm_fig = plt.figure(1)
        value_fig = plt.figure(2)
        self.swarm_plot = swarm_fig.add_subplot(111)
        self.value_plot = value_fig.add_subplot(111)
        self.value_plot.set_xlabel("iterations")
        self.value_plot.set_ylabel("f(gbest)")
        self.value_plot_domain = value_plot_domain
        self.swarm_plot_domain = swarm_plot_domain
        self.iterations_to_plot = iterations_to_plot
        self.func_str = func_str
        self.sleep = sleep

        if self.value_plot_domain:
            self.value_plot.axis(self.value_plot_domain)
        else:
            self.value_plot_max_y = - float("inf")
            self.value_plot_min_y = float("inf")

    def set_up_value_plot(self, label):
        """Sets the label of the value plot."""
        self.line, = self.value_plot.plot([], "-", label=label)
        handles, labels = self.value_plot.get_legend_handles_labels()
        self.value_plot.legend(handles=handles.append(self.line), loc=1)

    def update_plots(self, swarm, gbest, iteration):
        """Redraws swarm_plot and value_plot."""
        if iteration > self.iterations_to_plot:
            return
        self.swarm_plot.cla()
        if self.swarm_plot_domain:
            self.swarm_plot.axis(self.swarm_plot_domain)
        else:
            self.swarm_plot.axis(
                [i for sublist in swarm[0].domain for i in sublist]
                )
        colormap = plt.get_cmap("gist_rainbow")
        colors = len(swarm)
        for i in range(len(swarm)):
            color = colormap(float(i) / colors)
            self.swarm_plot.plot(
                swarm[i].variables[0], swarm[i].variables[1],
                marker="o", markersize=10, color=color)
            self.swarm_plot.quiver(
                swarm[i].variables[0] - swarm[i].velocity[0],
                swarm[i].variables[1] - swarm[i].velocity[1],
                swarm[i].velocity[0],
                swarm[i].velocity[1],
                angles="xy", scale_units="xy", scale=1, width=0.002,
                color=color, headlength=0, headwidth=1)
        self.swarm_plot.plot(
            gbest.variables[0],
            gbest.variables[1],
            marker="*", markersize=20, color="black",
            label="Global best")
        self.swarm_plot.legend(loc=1)
        self.swarm_plot.set_xlabel("x[0]")
        self.swarm_plot.set_ylabel("x[1]")

        if not self.value_plot_domain:
            # Adjust the y limits of the value plot
            self.value_plot_max_y = max(
                self.value_plot_max_y, gbest.best_value + 0.05)
            self.value_plot_min_y = min(
                self.value_plot_min_y, gbest.best_value - 0.05)
            self.value_plot.axis(
                [0, self.iterations_to_plot,
                 self.value_plot_min_y, self.value_plot_max_y]
                 )

        x = numpy.append(self.line.get_xdata(), iteration)
        y = numpy.append(self.line.get_ydata(), gbest.best_value)
        self.line.set_xdata(x)
        self.line.set_ydata(y)

        plt.title("Swarm during iteration {0}\n{1}\nGbest {2}\n value {3}".format(
            iteration, self.func_str, gbest.best_variables, gbest.best_value))

        plt.pause(self.sleep)

    def show(self):
        plt.show()
