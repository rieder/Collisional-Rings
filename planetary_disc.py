# coding: utf-8
import os,sys

import numpy as np

from scipy.stats import powerlaw

import matplotlib
matplotlib.use('Agg')
#matplotlib.use("Cairo")
import matplotlib.pyplot as plt

from amuse.community.rebound.interface import Rebound

from plotting import plot_interaction, plot_system

from amuse.units.units import named

import collections

try:
    from amuse.units import units
    MEarth  = units.MEarth
    REarth  = units.REarth
except:
    from usagi.units import units
    MEarth  = units.MEarth
    REarth  = units.REarth

rhoEarth = units.MEarth / (4./3. * np.pi * units.REarth**3)

MMoon = named('Lunar mass', 'M_Moon', 0.0123 * units.MEarth)#7.35e22 * units.kg
aR = named('Roche limit radius for lunar density', 'a_R',  2.9 * units.REarth)

from amuse.lab import *

def get_roche_limit_radius(
        rho,
        ):
    # Kokubo 2000 eq 2
    a_R = 2.456 * (rho.value_in(rhoEarth))**(-1./3.) | units.REarth
    return a_R

def particle_radius(
        m,
        rho,
        ):
    # Kokubo 2000 eq 1

    radius = (
            (m.value_in(units.MEarth))**(1./3.) * 
            (rho.value_in(rhoEarth)**(-1./3.))
            ) | units.REarth
    return radius



class Resolve_Encounters(object):
    def __init__(
            self, 
            encounters_0,
            encounters_1,
            primary         = None,
            convert_nbody   = None,
            G               = constants.G,
            epsilon_n       = 0.1,
            epsilon_t       = 1,
            f               = 0.0,#1.0,
            time            = 0.0 | units.yr,
            timestep        = 1.0 | units.hour,
            ):
        #first_is_most_massive = (
        #        encounters_0.mass ==
        #        encounters_0.mass.maximum(
        #            encounters_1.mass
        #            )
        #        )
        #encounters_A   = Particles()
        #encounters_B   = Particles()
        #encounters_A.add_particles(
        #        encounters_0[np.where(first_is_most_massive==True)[0]]
        #        )
        #encounters_A.add_particles(
        #        encounters_1[np.where(first_is_most_massive==False)[0]]
        #        )
        #encounters_B.add_particles(
        #        encounters_1[np.where(first_is_most_massive==True)[0]]
        #        )
        #encounters_B.add_particles(
        #        encounters_0[np.where(first_is_most_massive==False)[0]]
        #        )

        #order_by_mass = encounters_A.mass.argsort()
        #self.all_encounters_A = encounters_A[order_by_mass]
        #self.all_encounters_B = encounters_B[order_by_mass]
        self.all_encounters_A = encounters_0
        self.all_encounters_B = encounters_1

        self.primary            = primary

        self.particles_modified = Particles()
        self.particles_removed  = Particles()

        self.epsilon_n  = epsilon_n
        self.epsilon_t  = epsilon_t
        self.G          = G
        self.f          = f
        self.time       = time
        #TODO: use timestep to determine length of velocity vectors
        self.timestep   = timestep 

        #self.get_constants()
        #self.get_velocity_after_encounter()
        self.velocity_change_after_encounter()
        self.get_hill_radius()
        self.get_jacobi_energy()
        self.get_encounter_type()

        # Resolve. 
        # NOTE: need to take ongoing changes into account...
        # First: collisions that don't result in a merger
        # Then: mergers
        if len(self.colliding) > 0:
            self.resolve_rebounders()
        if len(self.merging) > 0:
            self.resolve_mergers()

    def velocity_change_after_encounter(
            self,
            ):
        A = self.all_encounters_A
        B = self.all_encounters_B

        if self.epsilon_t != 1.0:
            return -1
        r = B.position - A.position
        v = B.velocity - A.velocity
        n = r/r.lengths().reshape((len(r),1))

        v_n = (
                v[:,0]*n[:,0] +
                v[:,1]*n[:,1] +
                v[:,2]*n[:,2]
                ).reshape((len(n),1)) * n

        self.d_v_A =  (1+self.epsilon_n) * v_n * (B.mass / (A.mass+B.mass)).reshape((len(B),1))
        self.d_v_B = -(1+self.epsilon_n) * v_n * (A.mass / (A.mass+B.mass)).reshape((len(A),1))


    def _get_velocity_after_encounter(
            self,
            A,
            B,
            ):
        epsilon_n   = self.epsilon_n
        epsilon_t   = self.epsilon_t

        # Constants
        m_A = A.mass
        m_B = B.mass
        M   = m_A + m_B
        mu  = m_A * m_B / M
        r   = B.position - A.position # Distance between particle centres
        v_A = A.velocity
        v_B = B.velocity
        v   = v_B - v_A
        R_A = A.radius
        R_B = B.radius

        # Simplified case for spheres
        #alpha   = 5./2. * mu**-1 
        beta    = 2./7.#1/(1+alpha*mu)

        n_hat   = VectorQuantity(
                (
                    r / 
                    r.lengths().reshape((len(r),1))
                    ),
                units.none,
                )
        # Ignoring spin for now
        #sigma   = 
        u       = v# + sigma
        u_n     = (
                u[:,0] * n_hat[:,0] +
                u[:,1] * n_hat[:,1] +
                u[:,2] * n_hat[:,2]
                ).reshape((len(n_hat),1)) * n_hat
        u_t     = u - u_n

        u_n_prime   = (1+epsilon_n) * u_n
        u_t_prime   = beta * (1-epsilon_t) * u_t
        u_prime     = u_t_prime + u_n_prime

        v_A_prime   = v_A + u_prime * (m_B/M).reshape((len(m_B),1))
        v_B_prime   = v_B - u_prime * (m_A/M).reshape((len(m_A),1))

        return v_A_prime, v_B_prime

    def get_jacobi_energy(
            self,
            ):
        """
        Taken from Canup & Esposito (1995/1994), with cues from Kokubo, Ida &
        Makino (2000)
        """
        A = self.all_encounters_A
        B = self.all_encounters_B

        # Constants
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
        v_d = v_A - v_B
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

        x = (
                r[:,0] * x_hat[:,0] +
                r[:,1] * x_hat[:,1] +
                r[:,2] * x_hat[:,2]
                )
        z = (
                r[:,0] * z_hat[:,0] +
                r[:,1] * z_hat[:,1] +
                r[:,2] * z_hat[:,2]
                )

        Omega   = (
                v_orb[:,0] * y_hat[:,0] +
                v_orb[:,1] * y_hat[:,1] +
                v_orb[:,2] * y_hat[:,2]
                ) / (2*np.pi * r_orb.lengths())

        #v_A_prime, v_B_prime = self.get_velocity_after_encounter(A,B)
        #v_A_prime = v_A + self.d_v_A
        #v_B_prime = v_B + self.d_v_B
        #v_B_prime = self.get_velocity_after_encounter(A,B)
        #v_d_prime = v_A_prime - v_B_prime

        # Remember this is a potential, not really an energy
        # But since mass is always > 0, no problem.
        self.E_J = ( 
                0.5 * v_d.lengths_squared() - 
                1.5 * x**2 * Omega**2 + 
                0.5 * z**2 * Omega**2 - 
                self.G*M/r.lengths() +
                4.5 * self.radius_Hill**2 * Omega**2
                )

    def get_hill_radius(
            self,
            ):
        A = self.all_encounters_A
        B = self.all_encounters_B

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
        
        self.radius_Hill = (
                M / 
                (3 * self.primary.mass)
                )**(1./3) * r_orb.lengths()
        

    def get_encounter_type(
            self,
            energy_unit = units.erg,
            mass_unit   = units.kg,
            length_unit = units.km,
            time_unit   = units.s,
            ):

        A = self.all_encounters_A
        B = self.all_encounters_B

        interaction_includes_planet = (
                (self.primary.key == A.key) ^
                (self.primary.key == B.key)
                )

        dx  = A.x - B.x
        dy  = A.y - B.y
        dz  = A.z - B.z
        dvx = A.vx - B.vx
        dvy = A.vy - B.vy
        dvz = A.vz - B.vz

        #approaching = (0.| length_unit**2 * time_unit**-1) < (
        #        dx * dvx +
        #        dy * dvy +
        #        dz * dvz
        #        )

        jacobi_energy_negative = (
                self.E_J < (0 | energy_unit / mass_unit)
                )

        within_hill_radius = (
                (A.radius + B.radius) < (self.f * self.radius_Hill)
                )

        merging  = (
                interaction_includes_planet ^
                (
                    jacobi_energy_negative &
                    within_hill_radius
                    )
                )
        not_merging = (merging == False)

        self.merging = np.where(merging)[0]
        self.not_merging = np.where(not_merging)[0]
        self.colliding = np.where(not_merging)[0]
        #self.colliding  = np.where(not_merging & approaching)[0]

    def resolve_rebounders(
            self,
            move_particles = True,
            correct_for_multiple_collisions = False,
            ):
        
        A_original = self.all_encounters_A[self.not_merging]
        B_original = self.all_encounters_B[self.not_merging]
        A_modified = A_original.copy()
        B_modified = B_original.copy()

        A_modified.velocity += self.d_v_A[self.not_merging]
        B_modified.velocity += self.d_v_B[self.not_merging]

        ## Find all particles that are in this list more than once

        #allkeys = np.unique(np.append(A_modified.key,B_modified.key))
        #bins = allkeys.searchsorted(
        #        np.append(A_modified.key,B_modified.key)
        #        )
        #keys_appearing_more_than_once = allkeys[
        #        np.where(np.bincount(bins) > 1)
        #        ]

        #pairs_appearing_more_than_once = np.array([])
        #print "Pairs with particles in more than one collision: %i"%len(pairs_appearing_more_than_once)
        #for key in keys_appearing_more_than_once:
        #    pairs_appearing_more_than_once = np.append(
        #            pairs_appearing_more_than_once,
        #            np.where(A_modified.key == key),
        #            )
        #    pairs_appearing_more_than_once = np.append(
        #            pairs_appearing_more_than_once,
        #            np.where(B_modified.key == key),
        #            )
        #pairs_appearing_more_than_once = np.uint64(np.unique(
        #        pairs_appearing_more_than_once,
        #        ))

        #


        #if correct_for_multiple_collisions:
        #    for i in pairs_appearing_more_than_once:
        #        A_i = A_modified[i:i+1] # This makes it a particle set instead of a particle
        #        B_i = B_modified[i:i+1]
        #        A_i.velocity, B_i.velocity = \
        #                self.get_velocity_after_encounter(A_i, B_i)
        #        d_A = A_modified.select(lambda k: k == A_modified[i].key,["key"])
        #        d_A.velocity = A_modified[i].velocity
        #        d_B = B_modified.select(lambda k: k == B_modified[i].key,["key"])
        #        d_B.velocity = B_modified[i].velocity

        if move_particles:
            # Try to make sure the radii no longer overlap
            # This fails because we should be looking in a rotating frame... 
            
            # First, calculate the time that they have been overlapping
            #dt  = (
            #        (A.position-B.position).lengths() / 
            #        (A.velocity-B.velocity).lengths()
            #        )
            # Second, move them to the point of first collision
            m_A = A_modified.mass
            m_B = A_modified.mass
            M   = m_A + m_B

            r   = B_modified.position - A_modified.position
            #Distance the particles are overlapping:
            d   = r.lengths() - B_modified.radius - A_modified.radius

            n_hat   = VectorQuantity(
                    (
                        r / 
                        r.lengths().reshape((len(r),1))
                        ),
                    units.none,
                    )

            #Displacement post-velocity change:
            disp = (1+self.epsilon_n) * d.reshape((len(d),1)) * n_hat

            A_modified.position += (m_B/M).reshape((len(M),1)) * disp
            B_modified.position -= (m_A/M).reshape((len(M),1)) * disp

            # this may be an additional kick, but it seems fair enough...

        # Sync
        self.particles_modified.add_particles(A_modified)
        self.particles_modified.add_particles(B_modified)


    def resolve_mergers(
            self,
            ):
        # Need to do this in the right order!
        # 
        # First, need to be sure that m_1 >= m_2
        # 
        # Second, need to sort m_1 (small mergers first?)
        # 
        # Third, check for particles that occur more than once
        # and make sure to sync new values before continuing with merger
        #
        # Conserve position and velocity of center-of-mass
        # Combine total mass in the most massive particle
        # Choose the first one if masses are equal
        A = self.all_encounters_A
        B = self.all_encounters_B

        for i in range(len(self.merging)):
            index       = self.merging[i]
            if B[index].mass > A[index].mass:
                seed        = B[index]
                merge_with  = A[index]
            else:
                seed        = A[index]
                merge_with  = B[index]
            dist = (seed.position-merge_with.position).lengths()
            print "MERGER: p1 %s %s %s p2 %s %s %s EJ %s RH %s dist %s"%(
                    seed.key, seed.mass, seed.radius,
                    merge_with.key, merge_with.mass, merge_with.radius,
                    self.E_J[index], self.radius_Hill[index],
                    dist
                    )

            if merge_with.key in self.particles_removed.key:
                print "already merged!"
                break

            if seed.key in self.particles_removed.key:
                print "This should never happen!"
                print seed.key
                break

            if seed.key in self.particles_modified.key:
                # Particle already exists in modified form,
                # probably had a collision.
                # Use the modified form 
                # and remove it from the already-done list!
                seed = self.particles_modified.select(
                        lambda x: x == seed.key,["key"])[0].copy()
                self.particles_modified.remove_particle(seed)

            rho = seed.mass / (4/3. * np.pi * seed.radius**3)

            if merge_with.key in self.particles_modified.key:
                merge_with = self.particles_modified.select(
                        lambda x: x == merge_with.key, ["key"])[0].copy()
                self.particles_modified.remove_particle(merge_with)

            particles_to_merge = Particles()
            particles_to_merge.add_particle(seed)
            particles_to_merge.add_particle(merge_with)

            new_particle = seed.copy()
            new_particle.position   = particles_to_merge.center_of_mass()
            new_particle.velocity   = particles_to_merge.center_of_mass_velocity()
            new_particle.mass       = particles_to_merge.mass.sum()
            new_particle.radius     = particle_radius(new_particle.mass, rho)
            self.particles_removed.add_particle(merge_with)
            self.particles_modified.add_particle(new_particle)


class Planetary_Disc(object):
    """
    Class to resolve encounters and collisions in a disc around a planet.
    Collisions with the planet are also taken into account.
    """
    def __init__(self, options):
        """
        Initialise particles, identify subgroups.
        """
        self.options        = options
        convert_nbody       = self.options["converter"]
        self.f              = 0. if self.options["rubblepile"] else 1.0
        self.converter      = convert_nbody if convert_nbody != None else (
                nbody_system.nbody_to_si(
                    1|nbody_system.length, 
                    1|nbody_system.mass,
                    )
                )
        self.particles      = Particles()
        self.integrators    = []
        self.encounters     = []
        self.sync_attributes = ["mass", "radius", "x", "y", "z", "vx", "vy", "vz"]

        self.length_unit    = units.AU
        self.mass_unit      = units.kg
        self.speed_unit     = units.kms
        self.energy_unit    = units.erg
        self.time_unit      = units.yr
        self.particles.collection_attributes.nbody_length   = self.converter.to_si(1|nbody_system.length)
        self.particles.collection_attributes.nbody_mass     = self.converter.to_si(1|nbody_system.mass)

        self.model_time     = 0 | self.time_unit

    def exit_graceful(self):
        self.write_backup()
        exit()

    def write_backup(self, filename="continue.hdf5"):
        self.particles.collection_attributes.time = self.model_time
        self.particles.collection_attributes.timestep = self.integrator.model_time / self.integrator.parameters.timestep
        #self.particles.collection_attributes.grav_parameters = self.integrator.parameters
        write_set_to_file(self.particles,filename,"amuse")

    def evolve_model(self,time):
        while self.model_time < time:
            self.integrator.evolve_model(time)

            # Detect an error, save data in that case
            if self.integrator.particles[0].x.number == np.nan:
                self.exit_graceful()
            else:
                self.from_integrator_to_particles.copy()
                self.model_time = self.integrator.model_time

            if self.collision_detection.is_set():
                #if len(self.collision_detection.particles(0)) > 0:
                print "#Timesteps completed: %s"%(self.integrator.model_time / self.integrator.parameters.timestep)
                number_of_encounters = len(self.collision_detection.particles(0))

                m_before = self.integrator.particles.mass.sum()
                self.resolve_encounters()
                m_after = self.integrator.particles.mass.sum()

                if (
                        np.abs((m_after - m_before).value_in(units.MEarth)) > 
                        self.converter.to_si(
                            (1e-10|nbody_system.mass)
                            ).value_in(units.MEarth)
                        ):
                    print "Mass changed!", (m_after - m_before).as_quantity_in(units.MEarth)
                print "#Handled %i encounters this timestep"%(number_of_encounters)

    def define_subgroups(self):
        self.star       = self.particles.select(lambda x: x == "star", ["type"])
        self.planet     = self.particles.select(lambda x: x == "planet", ["type"])
        self.moon       = self.particles.select(lambda x: x == "moon", ["type"])
        self.disc       = self.particles.select(lambda x: x == "disc", ["type"])

    def add_particle(self, particle):
        self.particles.add_particle(particle)
        self.integrator.particles.add_particle(particle)
        self.define_subgroups()

    def add_particles(self, particles):
        self.particles.add_particles(particles)
        self.integrator.particles.add_particles(particles)
        self.define_subgroups()

    def remove_particle(self, particle):
        self.particles.remove_particle(particle)
        self.integrator.particles.remove_particle(particle)
        self.define_subgroups()
        
    def remove_particles(self, particles):
        self.particles.remove_particles(particles)
        self.integrator.particles.remove_particles(particles)
        self.define_subgroups()

    def add_integrator(self, integrator):
        self.integrator             = integrator
        self.collision_detection    = integrator.stopping_conditions.collision_detection
        self.collision_detection.enable()

        self.from_integrator_to_particles = \
                self.integrator.particles.new_channel_to(self.particles)
        self.from_particles_to_integrator = \
                self.particles.new_channel_to(self.integrator.particles)

    def resolve_encounters(
            self,
            ):
        #f   = 1.0 # fraction of the Hill radius

        resolution = Resolve_Encounters(
                self.collision_detection.particles(0).copy(),
                self.collision_detection.particles(1).copy(),
                primary = self.planet[0].copy(),
                time    = self.model_time,
                f       = self.f,
                )
        self.remove_particles(resolution.particles_removed)
        from_encounter_to_particles = \
                resolution.particles_modified.new_channel_to(self.particles)
        from_encounter_to_particles.copy_attributes(self.sync_attributes)
        self.from_particles_to_integrator.copy_attributes(self.sync_attributes)

        #plot_system(self.particles, "latest.png")

def main():
    options     = {}
    options["rubblepile"]   = False#True

    backupdir   = "./backup/"
    plotdir     = "./plots/"
    # Read the initial conditions file provided. This uses "Giant Impact" units.
    
    mass_unit   = 1|units.MEarth
    length_unit = get_roche_limit_radius(3.3|units.g * units.cm**-3)
    converter   = nbody_system.nbody_to_si(mass_unit, length_unit)
    options["converter"] = converter

    time = 0|units.yr
    if len(sys.argv) >= 2:
        filename = sys.argv[1]
        ext = filename.split('.')[-1]
        if ext == "txt":
            data = open(filename,"r").readlines()
            
            time = converter.to_si(float(data[0]) | nbody_system.time)
            nparticles = int(data[1])
            particles = Particles(nparticles)
            for n in range(nparticles):
                line = data[2+n].split()
                number = int(line[0])
                mass, radius, x, y, z, vx, vy, vz = map(float,line[1:])
                particles[n].number = number
                particles[n].mass = converter.to_si(mass|nbody_system.mass)
                particles[n].radius = converter.to_si(radius | nbody_system.length)
                particles[n].position = converter.to_si(
                        VectorQuantity(unit=nbody_system.length,array=[x,y,z])
                        )
                particles[n].velocity = converter.to_si(
                        VectorQuantity(unit=nbody_system.speed,array=[vx,vy,vz])
                        )
        
            particles.position -= particles.center_of_mass()
            particles.velocity -= particles.center_of_mass_velocity()
            
            particles[0].type   = "planet"
            particles[1:].type  = "disc"
            backupdir += filename.split('/')[-1][:-4] 
            plotdir   += filename.split('/')[-1][:-4] 
        elif ext == "hdf5":
            particles = read_set_from_file(filename, "amuse")
            backupdir += filename.split('/')[-1][:-5] 
            plotdir   += filename.split('/')[-1][:-5] 
        else:
            print "Unknown filetype"
            exit()

    else:
        particles = initial_particles(10000)
        write_set_to_file(particles,"this_run.hdf5","amuse",)

    if options["rubblepile"]:
        backupdir   += "-rubblepile"
        plotdir     += "-rubblepile"
    backupdir   += "/"
    plotdir     += "/"
    try:
        os.makedirs(backupdir)
        os.makedirs(plotdir)
    except:
        #FIXME make a new dir in this case, to prevent overwriting old files
        # use a datetime stamp
        print "#plotdir and/or backupdir already present"
        exit()

    particles[0].colour = "blue"
    particles[1:].colour = "black"

    #gravity = Hermite(converter)
    #gravity = ph4(converter)
    gravity = Rebound(converter)

    planetary_disc = Planetary_Disc(options)
    planetary_disc.add_integrator(gravity)
    planetary_disc.add_particles(particles)

    # Start up gravity code (Rebound) and simulate for a few Kepler times
    
    kepler_time = converter.to_si(
            2 * np.pi * 
            ( 
                (1|nbody_system.length)**3 / 
                ((1|nbody_system.mass) * nbody_system.G) 
                )**0.5
            )

    timestep_k2000 = (kepler_time/(2*np.pi))*(2**-9)
    
    gravity.parameters.timestep     = timestep_k2000
    gravity.parameters.integrator   = "whfast"
    #print gravity.parameters
    
    t_start         = time
    plot_time       = time
    backup_time     = time
    timestep        = 1.0 |units.minute 
    plot_timestep   = 1.0 |units.minute
    backup_timestep = 10. |units.minute
    t_end           = 10000. |units.hour 
    
    backup = 0
    plot = 0
    
    while time < t_end:
        if time >= plot_time:
            plot_system(
                    planetary_disc.particles,
                    "%s/plot-%05i.png"%(plotdir,plot),
                    )
            plot += 1
            plot_time += plot_timestep
        if time >= backup_time:
            planetary_disc.write_backup(filename="%s/savefile-%i.hdf5"%(backupdir,backup))
            backup += 1
            backup_time += backup_timestep
        if planetary_disc.model_time >= time:
            time += timestep
            print "#Now at time %s, evolving to %s (%i particles)"%(
                    gravity.model_time,time, 
                    len(gravity.particles),
                    )
        planetary_disc.evolve_model(time)


    gravity.stop()

if __name__ == "__main__":
    main()
