import sys
from amuse.lab import *
import matplotlib.pyplot as plt

from amuse.units.units import named

import numpy as np

aR = named('Roche limit radius for lunar density', 'a_R',  2.9 * units.REarth)

def plot_interaction(
        a, # particles before interaction
        b, # particles after interaction
        primary = False,
        figname="figure.png",
        ):

    # Find important axes:
    # N = normal
    # T = 90 deg from normal, pointing to primary

    # Constants
    A = a[0].copy()
    B = a[1].copy()
    m_A = A.mass
    m_B = B.mass
    M   = m_A + m_B
    r_A = A.position
    r_B = B.position
    r   = (
            r_A * m_A.reshape((len(m_A),1)) +
            r_B * m_B.reshape((len(m_B),1))
            ) / M.reshape((len(M),1))
    r_p = self.primary.position
    r_orb   = r - r_p

    v_A = A.velocity + self.d_v_A
    v_B = B.velocity + self.d_v_B
    v_c = (
            v_A * m_A.reshape((len(m_A),1)) +
            v_B * m_B.reshape((len(m_B),1))
            ) / M.reshape((len(M),1))
    v_d = v_B - v_A
    v_p = self.primary.velocity
    v_orb   = (v_c - v_p)

    # Derived
    x_hat   = VectorQuantity(
            (
                r_orb / 
                r_orb.lengths().reshape((len(r_orb),1))
                ),
            units.none,
            )
    v_orb_hat   = VectorQuantity(
            (
                v_orb / 
                v_orb.lengths().reshape((len(v_orb),1))
                ),
            units.none,
            )
    z_hat       = x_hat.cross(v_orb_hat)
    y_hat       = x_hat.cross(z_hat)

    c = [a,b]
    length_unit = units.REarth
    speed_unit = a.vx.unit
    # Rotate particles to these coordinates

    
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
        center=VectorQuantity([0,0,0],units.AU),
        plot_roche = True,
        time = 0|units.yr,
        ):
    xmin = [2.5, 0.]
    xmax = [2.5, 2.5]
    ymin = [2.5, 0.25]
    ymax = [2.5, 0.25]
    particles               = particles.copy()
    particles[0].colour     = "orange"
    particles[1].colour     = "red"
    particles[2:].colour    = "black"
    length_unit = units.AU#aR
    fig = plt.figure(figsize=(14,8), dpi=150)
    axes  = []
    axes.append(fig.add_subplot(1,2,1,aspect=1))
    axes.append(fig.add_subplot(1,2,2,aspect=1))
    maxmass = particles.mass.max()
    if center_on_most_massive:
        most_massive = particles.select(lambda x: x == maxmass,["mass"])[0]
        center = most_massive.position
    x = particles.x - center.x
    y = particles.y - center.y
    z = particles.z - center.z
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

    fig.canvas.draw()
    #FIXME Plot Roche radius
    #roche = plt.Circle(
    #        (
    #            (x_axes[0][0]).value_in(length_unit),
    #            (y_axes[0][0]).value_in(length_unit),
    #            ),
    #        2.9 * particles[0].radius.value_in(length_unit),
    #        facecolor="none",
    #        edgecolor="black",
    #        alpha = 0.5,
    #        )
    #axes[0].add_artist(roche)

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
                    #alpha       = 0.1,
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
                            #alpha       = 0.5,
                            )
                        )
                ax.add_artist(circles[-1])

    
        ax.set_xlim(-xmin[i],xmax[i])
        ax.set_ylim(-ymin[i],ymax[i])
        ax.set_xlabel("[%s]"%length_unit)
        ax.set_ylabel("[%s]"%length_unit)

        r       = particles.radius.value_in(length_unit)
        r[0]  = r[1]*0.5
        #r[2:] *= 3
        N       = len(particles)
        # Calculate radius in pixels :
        rr_pix  = (ax.transData.transform(np.vstack([r, r]).T) -
                  ax.transData.transform(np.vstack([np.zeros(N), np.zeros(N)]).T))
        rpix, _ = rr_pix.T

        # Calculate and update size in points:
        size_pt = (2*rpix/fig.dpi*72)**2
        scat.set_sizes(size_pt)
    fig.suptitle("time: %011.2f yr"%(time.value_in(units.yr)),fontsize=14)
    plt.savefig(plotname, dpi=fig.dpi)
    plt.close(fig)
    return



if __name__ == "__main__":
    particles = read_set_from_file(sys.argv[1],'amuse')
    plot_system(particles, "plot.png")
