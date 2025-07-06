from __future__ import annotations
from typing import TYPE_CHECKING
import math

if TYPE_CHECKING:
    from Particle import Particle

class FourVector:
    """Stores energy and momentum of a particle as a 4-vector.

    Attributes:
        energy (float):
            The energy in GeV.
        px (float):
            The momentum in the x-direction in GeV.
        py (float):
            The momentum in the y-direction in GeV.
        pz (float):
            The momentum in the z-direction in GeV.
    """

    def __init__(self, energy: float, px: float, py: float, pz: float) -> None:
        """Creates a 4-vector.
        
        Parameters:
            energy (float):
                The energy in GeV.
            px (float):
                The momentum in the x-direction in GeV.
            py (float):
                The momentum in the y-direction in GeV.
            pz (float):
                The momentum in the z-direction in GeV.
        """
        self.energy = energy
        self.px = px
        self.py = py
        self.pz = pz

    def __str__(self) -> str:
        """Turns the four vector into a string.
        
        Returns:
            str:
                The four vector as a string in the
                form (E, px, py, pz).
        """

        return f"({self.energy}, {self.px}, {self.py}, {self.pz})"

    def SumFourVectors(particles: list[Particle]) -> FourVector:
        """Sums the 4-vectors of a list of particles.

        Parameters:
            list[Particle]:
                A list of the detected particles to sum together.
        
        Returns:
            FourVector:
                The combined four vector.
        """
        
        # Creates a base four vector.
        combinedFourVector = FourVector(0, 0, 0, 0)
        
        # Adds all of the four vectors of the events to combine.
        for particle in particles:
            combinedFourVector += particle.fourVector

        return combinedFourVector

    # Note: In units of c = 1, the rest mass is the magnitude of the 4-vector.
    def Magnitude(self) -> float:
        """Gets the magnitude of the four vector.
        
        It is important to note that the magnitude of the 4-vector
        is E^2 - px^2 - py^2 - pz^2, which differs from normal vector magnitude.
        Also, since we are using units of c = 1, the magnitude of the 4-vector is
        the invariant mass in GeV.
        
        Returns:
            float:
                The magnitude of the 4-vector, equal to the invariant mass.
        """

        return math.sqrt(self.energy**2 - self.px**2 - self.py**2 - self.pz**2)
    
    def __add__(self, other: FourVector) -> FourVector:
        """Adds two 4-vectors together component-wise.
        
        Parameters:
            other (FourVector):
                The other 4-vector to add to the first one.
                
        Returns:
            FourVector:
                The combined 4-vector.
        """

        return FourVector(self.energy + other.energy, self.px + other.px, self.py + other.py, self.pz + other.pz)
    
    def toList(self) -> list[float]:
        """Converts the 4-vector into a list.
        
        Returns:
            list[float]:
                The 4-vector in list form.
        """

        return [self.energy, self.px, self.py, self.pz]