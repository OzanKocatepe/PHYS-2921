from Particle import Particle
from FourVector import FourVector
from typing import Callable

class Collision:
    """Contains all of the decay products from a single collision.

    Attributes:
        particles (list[Particle]):
            A list of all the particles created or detected (depending on
            whether this is simulated or experimental data) from a single
            collision, ordered by decreasing transverse momentum.
    """

    def __init__(self, particles: list[Particle]) -> None:
        """Creates an instance of a collision.
        
        Parameters:
            particles (list[Particle]):
                A list of all the particles belonging to this
                collision.
        """

        self.particles = particles
        # Orders the particles.
        self._OrderParticles()

    def GetCombinedFourVector(self) -> FourVector:
        """Gets the combined four vector of this collision.
        
        Returns:
            FourVector:
                The combined four vector of the colliison.
        """

        return FourVector.SumFourVectors(self.particles)

    def GetCombinedInvariantMass(self) -> float:
        """Gets the combined invariant mass of this collision.
        
        Returns:
            float:
                The combined invariant mass of the collision, in GeV.
        """

        return self.GetCombinedFourVector().Magnitude()
    
    def _OrderParticles(self, key: Callable[[Particle], any]=None, descendingOrder: bool=True) -> None:
        """Orders the particles of the collision in-place.

        By default, orders the particles first by pdg_id, and then
        by decreasing transverse momentum.

        Parameters:
            key (Callable[[Particle], any]):
                A function that takes in a particle as input
                and returns the desired attribute to sort by.
            descendingOrder (bool):
                Whether to order the list in descending order.
        """

        # If no key is given, uses the id and transverse momentum key.
        if key == None:
            key = Collision._idAndTransverseMomentumKey

        # Uses the sort function, with reverse=True to sort in descending order,
        # and the key to specify that the transverse momentum should be used as the
        # sorting criteria.
        self.particles.sort(reverse=descendingOrder, key=key)

    def _idAndTransverseMomentumKey(particle: Particle) -> tuple[int, float]:
        """The key used to sort particles by pdg_id, and then sort
        particles with the same id by transverse momentum.
        
        Parameters:
            particle (Particle):
                The particle to get the attribute of while sorting.

        Returns:
            int:
                The pdg_id of the particle.
            float:
                The transverse momentum of the particle.
        """

        return Collision._idKey(particle), Collision._TransverseMomentumKey(particle)

    def _idKey(particle: Particle) -> int:
        """The key used to sort particles by pdg_id.
        
        Parameters:
            particle (Particle):
                The particle to get the attribute of while sorting.

        Returns:
            int:
                The absolute value of the pdg_id of the particle.
        """

        return abs(particle.pdg_id)

    def _TransverseMomentumKey(particle: Particle) -> float:
        """The key used to order the particles by
        transverse momentum.

        Parameters:
            particle (Particle):
                The particle to get the attribute of while sorting.

        Returns:
            float:
                The transverse momentum of the particle.
        """

        return particle.GetTransverseMomentum()
    
    def Filter(self, ids: list[int]) -> None:
        """Filters the particles in the collision to only contain desired types of particles.
        
        The filtering is done in place, so nothing is returned.
        
        Parameters:
            ids (list[int]):
                The list of pdg_ids to keep. All other types of particles
                will be removed.
        """

        filteredParticles = []

        # Loops through all particles in the collision.
        for particle in self.particles:
            # Appends the particle to the new list
            # if its pdg_id is in the list of ids.
            if (particle.pdg_id in ids):
                filteredParticles.append(particle)

        # Sets the list of particles to be just the
        # filtered particles.
        self.particles = filteredParticles

        # Orders the particles. Since the particles are ordered upon
        # creation, the filtered list should still be ordered,
        # but better safe than sorry.
        self._OrderParticles()

    def CalculateAllDeltaR(self) -> dict[(int, int), float]:
        """Gets the delta R of every unique pair of particles
        in the collision.
        
        Returns:
            dict[(int, int), float]:
                A dictionary containing all the delta R values.
                The keys are of the form (i, j), where i is the index
                of the first particle in self.particles, and j is the index
                of the second particle. i and j necessarily satisfy
                1 <= i < j <= len(self.particles).
        """

        # Defines a dictionary to save the delta R values to.
        deltaRs = {}

        # Loops through every possible pair.
        for firstIndex in range(0, len(self.particles)):
            for secondIndex in range(firstIndex + 1, len(self.particles)):
                # Saves the delta R to the dictionary
                # under the key (firstIndex, secondIndex)
                deltaRs[(firstIndex, secondIndex)] = Particle.CalculateDeltaR(self.particles[firstIndex], self.particles[secondIndex])

        return deltaRs
    
    def GetFourVectors(self) -> list[FourVector]:
        """Gets a list of the four vectors of the particles in this collision,
        ordered in terms of decreasing transverse momentum.

        Returns:
            list[FourVector]:
                The ordered list of the particles' four vectors.
        """

        return [particle.fourVector for particle in self.particles]