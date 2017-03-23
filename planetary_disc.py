# coding: utf-8
import os,sys,shutil
import time as clocktime

import numpy as np

from scipy.stats import powerlaw

import matplotlib
matplotlib.use('Agg')
#matplotlib.use("Cairo")
import matplotlib.pyplot as plt

from amuse.community.rebound.interface import Rebound

from plotting import plot_interaction, plot_system

from amuse.units.units import named

from amuse.ext.orbital_elements import orbital_elements_for_rel_posvel_arrays

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
            f               = 0.0,
            time            = 0.0 | units.yr,
            ):
        self.all_encounters_A = encounters_0
        self.all_encounters_B = encounters_1
        self.number_of_collisions = len(self.all_encounters_A)

        self.primary            = primary

        self.particles_modified = Particles()
        self.particles_removed  = Particles()

        self.epsilon_n  = epsilon_n
        self.epsilon_t  = epsilon_t
        self.G          = G
        self.f          = f
        self.time       = time

        # Velocity changes only v_n -> can be done for ALL particles!
        self.update_velocities()
        # Should resolve collision immediately since the in-between state is unphysical
        # and may cause mergers within Roche radius
        self.resolve_rebounders()
        
        if self.f > 0.0:
            # These are used to determine if a merger will take place
            self.get_hill_radius()
            self.get_jacobi_energy()
        self.get_encounter_type()

        # Resolve. 
        # NOTE: need to take ongoing changes into account...
        # First: collisions that don't result in a merger
        # Then: mergers
        if len(self.merging) > 0:
            self.resolve_mergers()

    def update_velocities(
            self,
            ):
        A = self.all_encounters_A
        B = self.all_encounters_B

        if self.epsilon_t != 1.0:
            return -1
        r = B.position - A.position
        v = B.velocity - A.velocity
        n = r/r.lengths().reshape((self.number_of_collisions,1))

        v_n = (
                v[:,0]*n[:,0] +
                v[:,1]*n[:,1] +
                v[:,2]*n[:,2]
                ).reshape((len(n),1)) * n

        self.v_A_orig = A.velocity
        self.v_B_orig = B.velocity
        A.velocity += (1+self.epsilon_n) * v_n * (B.mass / (A.mass+B.mass)).reshape((self.number_of_collisions,1))
        B.velocity += -(1+self.epsilon_n) * v_n * (A.mass / (A.mass+B.mass)).reshape((self.number_of_collisions,1))

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
                r_A * m_A.reshape((self.number_of_collisions,1)) +
                r_B * m_B.reshape((self.number_of_collisions,1))
                ) / M.reshape((self.number_of_collisions,1))
        r_p = self.primary.position
        r_orb   = r - r_p

        v_A = A.velocity
        v_B = B.velocity
        v_c = (
                v_A * m_A.reshape((self.number_of_collisions,1)) +
                v_B * m_B.reshape((self.number_of_collisions,1))
                ) / M.reshape((self.number_of_collisions,1))
        v_d = v_B - v_A
        v_p = self.primary.velocity
        v_orb   = (v_c - v_p)

        # Derived
        x_hat   = VectorQuantity(
                (
                    r_orb / 
                    r_orb.lengths().reshape((self.number_of_collisions,1))
                    ),
                units.none,
                )
        v_orb_hat   = VectorQuantity(
                (
                    v_orb / 
                    v_orb.lengths().reshape((self.number_of_collisions,1))
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
                r_A * m_A.reshape((self.number_of_collisions,1)) +
                r_B * m_B.reshape((self.number_of_collisions,1))
                ) / M.reshape((self.number_of_collisions,1))
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
        if self.f > 0.0:
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
        else:
            merging = interaction_includes_planet

        not_merging = (merging == False)

        self.merging        = np.where(merging)[0]
        self.not_merging    = np.where(not_merging)[0]
        self.colliding      = self.not_merging#np.where(not_merging)[0]
        #self.colliding  = np.where(not_merging & approaching)[0]

    def resolve_rebounders(
            self,
            move_particles = True,
            correct_for_multiple_collisions = False,
            ):
        
        A = self.all_encounters_A
        B = self.all_encounters_B

        if move_particles:
            # Try to make sure the radii no longer overlap
            # This fails because we should be looking in a rotating frame... 
            
            m_A = A.mass
            m_B = B.mass
            M   = m_A + m_B

            r   = B.position - A.position
            #Distance the particles are overlapping:
            d   = r.lengths() - B.radius - A.radius

            n_hat   = VectorQuantity(
                    (
                        r / 
                        r.lengths().reshape((self.number_of_collisions,1))
                        ),
                    units.none,
                    )

            #Displacement post-velocity change:
            disp = (1+self.epsilon_n) * d.reshape((self.number_of_collisions,1)) * n_hat

            A.position +=  (m_B/M).reshape((self.number_of_collisions,1)) * disp
            B.position += -(m_A/M).reshape((self.number_of_collisions,1)) * disp

            # this may be an additional kick, but it seems fair enough...

        # Sync
        self.particles_modified.add_particles(A)
        self.particles_modified.add_particles(B)


    def resolve_mergers(
            self,
            ):
        # Conserve position and velocity of center-of-mass
        # Combine total mass in the most massive particle
        # Choose the first one if masses are equal
        A = self.all_encounters_A
        B = self.all_encounters_B

        # This has to be a for loop, since we have to account for multiple collisions with one object in one timestep.
        for i in range(len(self.merging)):
            index       = self.merging[i]
            if B[index].mass > A[index].mass:
                seed        = B[index]
                merge_with  = A[index]
            else:
                seed        = A[index]
                merge_with  = B[index]
            dist = (seed.position-merge_with.position).lengths()

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

        self.time_margin	= 0 | self.time_unit
        self.model_time         = 0 | self.time_unit
        self.kinetic_energy     = 0 | self.energy_unit
        self.potential_energy   = 0 | self.energy_unit
        self.disc_angular_momentum  = [0,0,0] | self.length_unit**2 * self.mass_unit * self.time_unit**-1

        self.timestep = self.options["timestep"]


    def exit_graceful(self):
        self.write_backup()
        exit()

    def write_backup(self, filename="continue.hdf5"):
        self.particles.collection_attributes.time = self.model_time
        if self.options["gravity"]=="Rebound":
            self.particles.collection_attributes.timestep = self.integrator.model_time / self.integrator.parameters.timestep
        if self.options["gravity"]=="Bonsai":
            self.particles.collection_attributes.timestep = self.integrator.model_time / self.integrator.parameters.timestep
        #self.particles.collection_attributes.grav_parameters = self.integrator.parameters
        write_set_to_file(self.particles,filename,"amuse")

    def evolve_model(self,time):
        if options["verbose"]>0:
            print "#Evolving to %s"%(time/self.timestep)
        if time > self.model_time:
            if options["verbose"]>0:
                print "#%s > %s, evolving..."%(
                        time/self.timestep, 
                        self.model_time/self.timestep,
                        )
        #print time/self.integrator.parameters.timestep, self.model_time/self.integrator.parameters.timestep
        #while (time - self.model_time) > 1e-10|units.s:#self.model_time < time:
            self.integrator.evolve_model(time + 0.0001*self.timestep)
            if options["verbose"]>0:
                print "#integrator now at %s"%(self.integrator.model_time/self.timestep)

            # Detect an error, save data in that case
            if self.integrator.particles[0].x.number == np.nan:
                self.exit_graceful()
            else:
                if options["verbose"]>0:
                    print "#Updating model"
                self.from_integrator_to_particles.copy()
                self.model_time             = self.integrator.model_time
                self.kinetic_energy         = self.integrator.kinetic_energy
                self.potential_energy       = self.integrator.potential_energy
                self.disc_angular_momentum  = self.disc.total_angular_momentum()

            if self.collision_detection.is_set():
                if options["verbose"]>0:
                    print "#Handling collisions (%i with %i)"%(
                        len(self.collision_detection.particles(0)),
                        len(self.collision_detection.particles(1)),
                        )
                if (
                        self.options["gravity"]=="Rebound" or
                        self.options["gravity"]=="Bonsai"
                        ):
                    if self.options["verbose"]>0:
                        print "#Timesteps completed: %s"%(self.integrator.model_time / self.timestep)
                number_of_encounters = len(self.collision_detection.particles(0))

                #m_before = self.integrator.particles.mass.sum()
                self.resolve_encounters()
                #m_after = self.integrator.particles.mass.sum()

                #if (
                #        np.abs((m_after - m_before).value_in(units.MEarth)) > 
                #        self.converter.to_si(
                #            (1e-10|nbody_system.mass)
                #            ).value_in(units.MEarth)
                #        ):
                #    print "#Mass changed!", (m_after - m_before).as_quantity_in(units.MEarth)
                if self.options["verbose"]>0:
                    print "#Handled %i encounters this timestep"%(number_of_encounters)
            if options["verbose"]>0:
                print "#Done"

    def define_subgroups(self):
        self.star       = self.particles.select(lambda x: x == "star", ["type"])
        self.planet     = self.particles.select(lambda x: x == "planet", ["type"])
        self.moon       = self.particles.select(lambda x: x == "moon", ["type"])
        self.disc       = self.particles.select(lambda x: x == "disc", ["type"])

    def add_particle(self, particle):
        self.add_particles(particle.as_set())
        #self.particles.add_particle(particle)
        #self.integrator.particles.add_particle(particle)
        #self.define_subgroups()

    def add_particles(self, particles):
        self.particles.add_particles(particles)
        self.integrator.particles.add_particles(particles)
        self.define_subgroups()

    def remove_particle(self, particle):
        self.remove_particles(particle.as_set())
        #self.particles.remove_particle(particle)
        #self.integrator.particles.remove_particle(particle)
        #self.define_subgroups()
        
    def remove_particles(self, particles):
        if options["verbose"]>0:
            print "#Removing %i particles"%(len(particles))
        #from_encounter_to_particles = \
        #        particles.new_channel_to(self.particles)
        #from_encounter_to_particles.copy_attributes(self.sync_attributes)
        #self.from_particles_to_integrator.copy_attributes(self.sync_attributes)
        self.particles.remove_particles(particles)
        self.integrator.particles.remove_particles(particles)
        self.define_subgroups()

    def add_integrator(self, integrator):
        self.integrator             = integrator
        self.collision_detection    = integrator.stopping_conditions.collision_detection
        try:
            self.integrator_timestep	= integrator.parameters.timestep
            self.time_margin		= 0.5 * self.integrator_timestep
        except:
            self.integrator_timestep	= False
        if not options["disable_collisions"]:
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

def main(options):
    now = clocktime.strftime("%Y%m%d%H%M%S")

    # Read the initial conditions file provided. This uses "Giant Impact" units.
    
    mass_unit   = options["unit_mass"]
    length_unit = options["unit_length"]
    converter   = nbody_system.nbody_to_si(1|mass_unit, 1|length_unit)
    options["converter"] = converter

    time        = options["time_start"]
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
            rundir = "./runs/" + filename.split('/')[-1][:-4] 
        elif ext == "hdf5":
            particles = read_set_from_file(filename, "amuse")
            rundir = "./runs/" + filename.split('/')[-1][:-5] 
        else:
            print "Unknown filetype"
            exit()

    else:
        particles = initial_particles(10000)
        write_set_to_file(particles,"this_run.hdf5","amuse",)

    
    rundir += "-%s-%s"%(
            now,
            options["gravity"],
            )
    if options["rubblepile"]:
        rundir   += "-rubblepile"

    backupdir   = rundir + "/backups"
    plotdir     = rundir + "/plots"

    try:
        os.makedirs(rundir)
        os.makedirs(backupdir)
        os.makedirs(plotdir)
        shutil.copy(sys.argv[0],rundir)
    except:
        #FIXME make a new dir in this case, to prevent overwriting old files
        # use a datetime stamp
        print "#directories already present"
        exit()

    particles[0].colour = "blue"
    particles[1:].colour = "black"
    
    kepler_time = converter.to_si(
            2 * np.pi * 
            ( 
                (1|nbody_system.length)**3 / 
                ((1|nbody_system.mass) * nbody_system.G) 
                )**0.5
            )

    converter_earthunits = nbody_system.nbody_to_si(1|units.MEarth,1|units.REarth)
    #time_unit   = kepler_time.value_in(units.s) * units.s
    #energy_unit = units.MEarth * units.REarth**2 * time_unit**-2
    #print (1|units.s).value_in(time_unit)
    #print (1|units.kg).value_in(mass_unit)
    #print (1|units.m).value_in(length_unit)
    #print (1|units.erg).value_in(energy_unit)

    timestep_k2000 = (kepler_time/(2*np.pi))*(2**-9)
    
    # Start up gravity code 
    if options["gravity"] == "Rebound":
        gravity = Rebound(converter,redirection="none")
        gravity.parameters.timestep     = timestep_k2000
        gravity.parameters.integrator   = options["integrator"]
        #gravity.parameters.integrator   = "leapfrog"
        #gravity.parameters.opening_angle2 = 0.5
        gravity.parameters.solver = "compensated"
        #gravity.parameters.solver = "tree"
        #gravity.parameters.boundary = "open"
        #gravity.parameters.boundary_size    = 10|units.AU
        if options["whfast_corrector"]:
            gravity.parameters.whfast_corrector = options["whfast_corrector"]
    elif options["gravity"] == "Bonsai":
        gravity = Bonsai(converter)
        gravity.parameters.timestep     = timestep_k2000
    elif options["gravity"] == "ph4":
        if options["use_gpu"]:
            gravity = ph4(converter, mode="gpu")
        else:
            gravity = ph4(converter)
    elif options["gravity"] == "phigrape":
        if options["use_gpu"]:
            gravity = PhiGRAPE(converter, mode="gpu")
        else:
            gravity = PhiGRAPE(converter)
    elif options["gravity"] == "Hermite":
        gravity = Hermite(converter, number_of_workers=6)
    else:
        print "Unknown gravity code"
        exit()
    gravity.parameters.epsilon_squared  = (particles[-1].radius)**2
    print gravity.parameters
    options["timestep"] = timestep_k2000

    planetary_disc = Planetary_Disc(options)
    planetary_disc.add_integrator(gravity)
    planetary_disc.add_particles(particles)

    t_start         = time
    plot_time       = time
    backup_time     = time
    timestep        = timestep_k2000#options["timestep"]
    plot_timestep   = options["timestep_plot"]
    backup_timestep = options["timestep_backup"]
    t_end           = options["time_end"]
    
    backup = 0
    plot = 0
    
    log_time                = VectorQuantity([],units.s)
    log_kinetic_energy      = VectorQuantity([],units.erg)
    log_potential_energy    = VectorQuantity([],units.erg)
    log_angular_momentum    = VectorQuantity([],units.AU**2 * units.MEarth * units.yr**-1)

    log = open(rundir+"/log.txt",'w')
    log.write("#1 time   = %s\n"%(converter_earthunits.to_si(1|nbody_system.time)))
    log.write("#1 length = %s\n"%(converter_earthunits.to_si(1|nbody_system.length)))
    log.write("#1 mass   = %s\n"%(converter_earthunits.to_si(1|nbody_system.mass)))
    log.write("#1 energy = %s\n"%(converter_earthunits.to_si(1|nbody_system.energy)))
    log.write("#Time N E_kin_G E_kin_A E_pot l2 l2_disc M_disc a_mean a_sigma e_mean e_sigma inc_mean inc_sigma\n")
    log.write("#%s n %s %s %s %s %s %s %s %s\n"%(
        units.s,
        nbody_system.energy,#s.erg,
        nbody_system.energy,#s.erg,
        nbody_system.energy,#s.erg,
        (units.REarth**2 * units.MEarth * units.day**-1)**2,
        (units.REarth**2 * units.MEarth * units.day**-1)**2,
        units.MEarth,
        units.REarth,
        units.REarth,
        )
        )
    log.flush()

    time += timestep_k2000
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
        if (time - planetary_disc.model_time) <= 0.5 * timestep:
            if options["verbose"]>0:
                print "#Increasing timestep: %s - %s <= 0.5"%(
                        planetary_disc.model_time / planetary_disc.integrator.parameters.timestep, 
                        time / planetary_disc.integrator.parameters.timestep,
                        )
            time += timestep
            #log_time.append(planetary_disc.model_time)
            #log_kinetic_energy.append(planetary_disc.kinetic_energy)
            #log_potential_energy.append(planetary_disc.potential_energy)
            #log_angular_momentum.append(planetary_disc.disc_angular_momentum)
            kinetic_energy = planetary_disc.kinetic_energy
            potential_energy = planetary_disc.potential_energy
            angular_momentum = planetary_disc.particles.total_angular_momentum()
            disc_angular_momentum = planetary_disc.disc.total_angular_momentum()
            semimajor_axis, eccentricity, true_anomaly,inc, long_asc_node, arg_per_mat = orbital_elements_for_rel_posvel_arrays(
                    planetary_disc.disc.position - planetary_disc.planet.position, 
                    planetary_disc.disc.velocity - planetary_disc.planet.velocity, 
                    planetary_disc.planet.mass,#total_masses, 
                    G=constants.G,
                    )
            #FIXME kinetic energy per particle
            #FIXME angular momentum per particle

            #print "Time %s N %i E_kin %s E_pot %s angmom %s %s %s"%(
            log.write("%s %i %s %s %s %s %s %s %s %s %s %s %s %s\n"%(
                    planetary_disc.model_time.value_in(units.s), 
                    len(planetary_disc.particles),
                    converter_earthunits.to_nbody(kinetic_energy).value_in(nbody_system.energy),
                    converter_earthunits.to_nbody(planetary_disc.disc.kinetic_energy()).value_in(nbody_system.energy),
                    converter_earthunits.to_nbody(potential_energy).value_in(nbody_system.energy),
                    (
                        angular_momentum[0]**2 + 
                        angular_momentum[1]**2 + 
                        angular_momentum[2]**2
                        ).value_in(units.REarth**4 * units.MEarth**2 * units.day**-2),
                    (
                        disc_angular_momentum[0]**2 + 
                        disc_angular_momentum[1]**2 + 
                        disc_angular_momentum[2]**2
                        ).value_in(units.REarth**4 * units.MEarth**2 * units.day**-2),
                    planetary_disc.disc.mass.sum().value_in(units.MEarth),
                    semimajor_axis.mean().value_in(units.REarth), 
                    semimajor_axis.std().value_in(units.REarth),
                    eccentricity.mean(), 
                    eccentricity.std(),
                    inc.mean(), 
                    inc.std(),
                    )
                    )
            log.flush()
        else:
            if options["verbose"]>0:
                print "#Not increasing timestep: %s - %s > 0.5"%(
                        planetary_disc.model_time / planetary_disc.integrator.parameters.timestep, 
                        time / planetary_disc.integrator.parameters.timestep,
                        )
        planetary_disc.evolve_model(time)

    gravity.stop()
    log.close()

if __name__ == "__main__":
    options     = {}
    options["verbose"]          = 0
    options["rubblepile"]       = True
    options["gravity"]          = "Rebound"
    options["integrator"]       = "ias15"
    options["whfast_corrector"] = 0
    options["use_gpu"]          = False
    options["time_start"]       = 0. | units.yr
    options["time_end"]         = 10000. |units.hour 
    options["timestep"]         = 1. |units.minute 
    options["timestep_plot"]    = 1. |units.minute 
    options["timestep_backup"]  = 60. |units.minute 
    options["unit_mass"]        = units.MEarth
    options["disable_collisions"]   = False
    options["unit_length"]      = get_roche_limit_radius(3.3|units.g * units.cm**-3).value_in(units.REarth) * units.REarth
    main(options)
