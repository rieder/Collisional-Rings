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

    def handle_collisions(self, particles, primary_indices, secondary_indices):
        d_pos = VectorQuantity(
                np.zeros(len(particles)*3).reshape(len(particles),3),
                particles.position.unit,
                )
        d_vel = VectorQuantity(
                np.zeros(len(particles)*3).reshape(len(particles),3),
                particles.velocity.unit,
                )

        # This loop could be run with many threads...
        primaries   = particles[primary_indices]
        secondaries = particles[secondary_indices]
            
        relative_position   = secondaries.position - primaries.position
        distance            = relative_position.lengths()
        relative_velocity   = secondaries.velocity - primaries.velocity
        surface_distance    = distance - primaries.radius - secondaries.radius
        total_mass          = primaries.mass + secondaries.mass
    
        normal  = relative_position / distance.reshape((len(primaries),1))
        nx      = normal[:,0]
        ny      = normal[:,1]
        nz      = normal[:,2]
        normal_velocity = (
                relative_velocity[:,0]*nx + 
                relative_velocity[:,1]*ny + 
                relative_velocity[:,2]*nz
                )
        
        pshift = VectorQuantity(
                np.zeros(len(primaries)*3).reshape(len(primaries),3),
                (particles.position.unit / particles.mass.unit),
                )
        vshift = VectorQuantity(
                np.zeros(len(primaries)*3).reshape(len(primaries),3),
                (particles.velocity.unit / particles.mass.unit),
                )
        pshift[:,0] = (1./total_mass * surface_distance * nx)
        pshift[:,1] = (1./total_mass * surface_distance * ny)
        pshift[:,2] = (1./total_mass * surface_distance * nz)
        vshift[:,0] = (1./total_mass * (1.0+self.epsilon_n) * normal_velocity * nx)
        vshift[:,1] = (1./total_mass * (1.0+self.epsilon_n) * normal_velocity * ny)
        vshift[:,2] = (1./total_mass * (1.0+self.epsilon_n) * normal_velocity * nz)
        d_pos[primary_indices] +=  secondaries.mass.reshape((len(primaries),1)) * pshift
        d_vel[primary_indices] +=  secondaries.mass.reshape((len(primaries),1)) * vshift
        d_pos[secondary_indices] += -primaries.mass.reshape((len(primaries),1)) * pshift
        d_vel[secondary_indices] += -primaries.mass.reshape((len(primaries),1)) * vshift

        return d_pos, d_vel
