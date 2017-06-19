from amuse.ext.orbital_elements import (
        rel_posvel_arrays_from_orbital_elements,
        orbital_period_to_semimajor_axis
        )
from amuse.lab import (
        units, constants,
        write_set_to_file,
        Particle, Particles,
        )
import numpy as np
from numpy import pi


if __name__ in ("__main__"):
    # Important properties
    Mstar = 1 | units.MSun
    Mplanet = 80 | units.MJupiter
    Mdisc = 10 | units.MEarth
    T = 11 | units.yr
    e = 0.65
    disc_density = 3.3 | units.g * units.cm**-3

    # Star
    star = Particle()
    star.mass = Mstar
    star.type = "star"
    star.radius = 1 | units.RSun

    # Planet orbit
    a = orbital_period_to_semimajor_axis(
            T, Mstar, Mplanet+Mdisc, G=constants.G)
    inc = 0 | units.rad
    TA = pi | units.rad  # start at apocentre
    LOTAN = 0 | units.rad
    AOP = 0 | units.rad
    planetpos, planetvel = rel_posvel_arrays_from_orbital_elements(
            Mstar, Mplanet, a, eccentricity=e,
            true_anomaly=TA, inclination=inc,
            longitude_of_the_ascending_node=LOTAN,
            argument_of_periapsis=AOP,
            G=constants.G
            )

    planet = Particle()
    planet.position = planetpos
    planet.velocity = planetvel
    planet.mass = Mplanet
    planet.type = "planet"
    planet.radius = 4 | units.RJupiter

    # Disc properties
    N = 100000
    discmass = np.ones(N) * Mdisc/N
    discamin = 0.2 | units.AU
    discamax = 0.3 | units.AU
    disca = discamin + (np.random.random(N) * (discamax-discamin))
    discemin = 0.
    discemax = 0.05
    disce = discemin + (np.random.random(N) * (discemax-discemin))
    discinc = pi + 0.5 * disce | units.rad
    discLOTAN = 2 * pi * np.random.random(N) | units.rad
    discAOP = 2 * pi * np.random.random(N) | units.rad
    discTA = 2 * pi * np.random.random(N) | units.rad

    disc = Particles(N)

    discpos, discvel = rel_posvel_arrays_from_orbital_elements(
            Mplanet, discmass, disca, eccentricity=disce,
            true_anomaly=discTA, inclination=discinc,
            longitude_of_the_ascending_node=discLOTAN,
            argument_of_periapsis=discAOP,
            G=constants.G
            )

    disc.position = discpos + planetpos
    disc.velocity = discvel + planetvel
    disc.mass = discmass
    disc.radius = (
            disc.mass
            / disc_density
            / (4./3.)
            / pi
            )**(1./3.)
    disc.type = "disc"

    particles = Particles()
    particles.add_particle(star)
    particles.add_particle(planet)
    particles.add_particles(disc)

    particles.position -= particles.center_of_mass()
    particles.velocity -= particles.center_of_mass_velocity()

    write_set_to_file(
            particles,
            "disc_retro.hdf5",
            "amuse",
            )
