import sys
import numpy

from amuse.units import units
from amuse.datamodel import Particles
from amuse.support.exceptions import AmuseException

from amuse.units.quantities import VectorQuantity
import numpy as np


class BotsRots(object):
    """
    Resolves collisions between particles by treating them as somewhat inelastic
    collisions. The degree of rebounding is controlled by the epsilon_n parameter.
    
    """
    
    stellar_evolution_code_required = False
    gravity_code_required = False
    
    def __init__(self, epsilon_n=0.1):
        if 0 <= epsilon_n <= 1:
            self.epsilon_n = epsilon_n
        else:
            raise AmuseException("epsilon_n must be in the range [0, 1]")

    def handle_collision(self, particles, primary_indices, secondary_indices):
        d_pos = VectorQuantity(
                np.zeros(len(particles)*3).reshape(len(particles),3),
                particles.position.unit,
                )
        d_vel = VectorQuantity(
                np.zeros(len(particles)*3).reshape(len(particles),3),
                particles.velocity.unit,
                )

        for number in range(len(primary_indices)):
            i   = primary_indices[number]
            j   = secondary_indices[number]
            primary     = particles[i]
            secondary   = particles[j]

            relative_position   = secondary.position - primary.position
            distance            = relative_position.length()
            relative_velocity   = secondary.velocity - primary.velocity
            surface_distance    = distance - primary.radius - secondary.radius
    
            if (
                    surface_distance <= 0|distance.unit
                    ):
                normal  = relative_position / distance
                nx      = normal[0]
                ny      = normal[1]
                nz      = normal[2]
                normal_velocity = (relative_velocity[0]*nx + relative_velocity[1]*ny + relative_velocity[2]*nz)
            
                total_mass  = primary.mass + secondary.mass
                pshift  = []
                vshift  = []
                pshift.append(1./total_mass * surface_distance * nx)
                pshift.append(1./total_mass * surface_distance * ny)
                pshift.append(1./total_mass * surface_distance * nz)
                vshift.append(1./total_mass * (1.0+self.epsilon_n) * normal_velocity * nx)
                vshift.append(1./total_mass * (1.0+self.epsilon_n) * normal_velocity * ny)
                vshift.append(1./total_mass * (1.0+self.epsilon_n) * normal_velocity * nz)
                for x in range(3):
                    d_pos[i,x] +=  secondary.mass * pshift[x]
                    d_vel[i,x] +=  secondary.mass * vshift[x]
                    d_pos[j,x] += -primary.mass * pshift[x]
                    d_vel[j,x] += -primary.mass * vshift[x]
        return d_pos, d_vel

