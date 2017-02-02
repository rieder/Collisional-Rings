import sys
from amuse.lab import *
import matplotlib.pyplot as plt

from amuse.units.units import named

import numpy as np

aR = named('Roche limit radius for lunar density', 'a_R',  2.9 * units.REarth)

def plot_interaction(
        a,
        b,
        figname="figure.png"
        ):
    c = [a,b]
    length_unit = units.REarth
    speed_unit = a.vx.unit
    
    fig = plt.figure(figsize=(11,6))

    plot_radius = max(
            2*(
            a[0].position - 
            a[1].position).length().value_in(length_unit),
            a.radius.max().value_in(length_unit)
            )
    plot_center = a.center_of_mass().value_in(length_unit)
    
    rocks = []
    for i in range(2):
        ax = fig.add_subplot(1,2,1+i, aspect=1)

        ax.scatter(
                c[i].x.value_in(length_unit),
                c[i].y.value_in(length_unit),
                s=1,#c[i].radius.value_in(length_unit),
                edgecolors=None,
                facecolors="black",
                )
        ax.quiver(
                c[i].x.value_in(length_unit),
                c[i].y.value_in(length_unit),
                c[i].vx.value_in(speed_unit),
                c[i].vy.value_in(speed_unit),
                facecolors="grey",
                )
        ax.set_xlim(
                plot_center[0] - plot_radius,
                plot_center[0] + plot_radius,
                )
        ax.set_ylim(
                plot_center[1] - plot_radius,
                plot_center[1] + plot_radius,
                )

        for p in c[i]:
            rocks.append(
                    plt.Circle(
                        (
                            (p.x).value_in(length_unit), 
                            (p.y).value_in(length_unit),
                            ),
                        p.radius.value_in(length_unit),
                        color = 'black',
                        alpha = 0.5,
                        )
                    )
            ax.add_artist(rocks[-1])

    plt.savefig(figname)
    plt.close(fig)

def plot_system(
        particles, 
        plotname, 
        scatter=True,
        center_on_most_massive=True,
        plot_roche = True,
        ):
    particles               = particles.copy()
    particles[0].colour     = "blue"
    particles[1:].colour    = "black"
    length_unit = aR
    fig = plt.figure(figsize=(14,8))
    axes  = []
    axes.append(fig.add_subplot(1,2,1,aspect=1))
    axes.append(fig.add_subplot(1,2,2,aspect=1))
    maxmass = particles.mass.max()
    most_massive = particles.select(lambda x: x == maxmass,["mass"])[0]
    x = particles.x - most_massive.x
    y = particles.y - most_massive.y
    z = particles.z - most_massive.z
    r = (x**2 + y**2)**0.5
    #r = (particles.position - most_massive.position).lengths()
    x_axes = [x,r]
    y_axes = [y,z]
    minmax = 1 + np.floor(
            max(
                -x_axes[0].min().value_in(length_unit),
                x_axes[0].max().value_in(length_unit),
                -y_axes[0].min().value_in(length_unit),
                y_axes[0].max().value_in(length_unit),
                )
            )
    xmin = [2., 0.]
    xmax = [2., 3.]
    ymin = [2., 0.5]
    ymax = [2., 0.5]

    fig.canvas.draw()
    #FIXME Plot Roche radius
    for i in range(len(axes)):
        ax = axes[i]

        if scatter:
            scat = ax.scatter(
                    (x_axes[i]).value_in(length_unit),
                    (y_axes[i]).value_in(length_unit),
                    marker      = 'o',
                    s           = 1,
                    edgecolors  = "none",
                    facecolors  = particles.colour,
                    alpha       = 0.5,
                    )
        else:
            circles = []
            for j in range(len(particles)):
                circles.append(
                        plt.Circle(
                            (
                                (x_axes[i][j]).value_in(length_unit),
                                (y_axes[i][j]).value_in(length_unit),
                                ),
                            particles[j].radius.value_in(length_unit),
                            facecolor   = particles[j].colour,
                            edgecolor   = "none",
                            alpha       = 0.5,
                            )
                        )
                ax.add_artist(circles[-1])

    
        ax.set_xlim(-xmin[i],xmax[i])
        ax.set_ylim(-ymin[i],ymax[i])
        ax.set_xlabel("[%s]"%length_unit)
        ax.set_ylabel("[%s]"%length_unit)

        r       = particles.radius.value_in(length_unit)
        N       = len(particles)
        # Calculate radius in pixels :
        rr_pix  = (ax.transData.transform(np.vstack([r, r]).T) -
                  ax.transData.transform(np.vstack([np.zeros(N), np.zeros(N)]).T))
        rpix, _ = rr_pix.T

        # Calculate and update size in points:
        size_pt = (2*rpix/fig.dpi*72)**2
        scat.set_sizes(size_pt)
    plt.savefig(plotname)
    plt.close(fig)
    return



if __name__ == "__main__":
    particles = read_set_from_file(sys.argv[1],'amuse')
    plot_system(particles, "plot.png")
