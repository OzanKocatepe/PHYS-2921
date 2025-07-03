from __future__ import annotations
import math

from FourVector import FourVector

class Particle:
    """A single particle detection.
 
    Attributes:
        pdg_id (int):
            The pdg_id of the particle.
        fourVector (FourVector):
            The 4-vector for this particle, with units of c = 1.
    """

    def __init__(self, pdg_id: int, energy: float, px: float, py: float, pz: float) -> None:
        """Creates an instance of a detected particle.
        
        Parameters:
            pdg_id (int):
                The pdg_id of the particle.
            energy (float):
                The energy of the particle in GeV.
            px (float):
                The particle's momentum in the x-direction in GeV.
            py (float):
                The particle's momentum in the y-direction in GeV.
            pz (float):
                The particle's momentum in the z-direction in GeV.
        """
        
        self.pdg_id = pdg_id
        self.fourVector = FourVector(energy, px, py, pz)

    def __str__(self) -> str:
        """Outputs the information stored in the instance as a string.
        
        Returns:
            str:
                A string which contains all of the information stored in
                this instance, formatted as a string.
        """

        return f"[pdg_id={self.pdg_id}, fourVector={self.fourVector}]"
    
    def CalculateDeltaR(event1: Particle, event2: Particle) -> float:
        """Calculates the Delta R between two particles. This property
        is Lorentz-invariant.
        
        Parameters:
            event1 (Particle):
                The first particle.
            event (Particle):
                The second particle.
                
        Returns:
            float:
                The delta R value between the two particles.
        """
 
        deltaEta = event1.GetPsuedorapidity() - event2.GetPsuedorapidity()
        deltaPhi = event1.GetPhi() - event2.GetPhi()
        deltaR = math.sqrt(deltaEta**2 + deltaPhi**2)

        return deltaR

    def GetMomentumMagnitude(self) -> float:
        """Gets the magnitude of the particle's momentum.
        
        Returns:
            float:
                The magnitude of the particle's momentum.
        """

        v = self.fourVector
        return math.sqrt(v.px**2 + v.py**2 + v.pz**2)

    def GetTransverseMomentum(self) -> float:
        """Calculates the transverse momentum of the particle, which is
        defined as the momentum in the x-y plane, using ATLAS coordinate
        system.
        
        Returns:
            float:
                The particle's transverse momentum.
        """

        v = self.fourVector
        return math.sqrt(v.px**2 + v.py**2)
    
    def GetPsuedorapidity(self) -> float:
        """Calculates the particle's psuedorapidity.
        
        The psuedorapidity is -ln(tan(theta / 2)), where
        theta is the angle from the z- (or beam-) axis
        using the ATLAS coordinate system.
        
        The difference between the pseudorapidity of two particles
        is Lorentz-invariant.
        
        Returns:
            float:
                The particle's psuedorapidity, or 0 if the particle is stationary.
        """

        try:
            # Uses the simplest formula for pseudorapidity.
            return math.atanh(self.fourVector.pz / self.GetMomentumMagnitude())
        except:
            # If the magnitude of the momentum is 0, then we have 0 / 0, which
            # we simplify to give a psuedorapidity of just 0.
            return 0

    def GetPhi(self) -> float:
        """Gets phi, which is the angle of the particle in the x-y plane,
        using the ATLAS coordinate system.

        Returns:
            float:
                The particle's angle in the x-y plane.
        """

        # Checks if the denominator is 0. If so, our input to
        # atan will be +/- infinity, so we'll return +/- pi/2.
        if (self.fourVector.px == 0):
            return math.pi * math.sign(self.fourVector.py) / 2

        # Uses the formula for phi given by Harish.
        return math.atan(self.fourVector.py / self.fourVector.px)
    
    def GetRestMass(self) -> float:
        """Gets the rest mass of the particle.
        
        Returns:
            float: The particle's rest mass in GeV.
        """

        return self.fourVector.Magnitude()

    def GetEnergy(self) -> float:
        """Gets the energy of the particle.
        
        Returns:
            float: The particle's energy in GeV.
        """

        return self.fourVector.energy

    def GetXMomentum(self) -> float:
        """Gets the particle's momentum in the x-direction,
        using the ATLAS coordinate system.
        
        Returns:
            float:
                The particle's x-momentum in GeV.
        """

        return self.fourVector.px
    
    def GetYMomentum(self) -> float:
        """Gets the particle's momentum in the y-direction,
        using the ATLAS coordinate system.
        
        Returns:
            float:
                The particle's y-momentum in GeV.
        """

        return self.fourVector.py
    
    def GetZMomentum(self) -> float:
        """Gets the particle's momentum in the z-direction,
        using the ATLAS coordinate system.
        
        Returns:
            float:
                The particle's z-momentum in GeV.
        """

        return self.fourVector.pz