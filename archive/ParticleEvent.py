from __future__ import annotations
import math
from FourVector import FourVector

class ParticleEvent:
    """A single particle detection.
 
    Attributes:
        eventGroup: int
            The event group in which this detection occured. Used to group the ParticleEvents into larger event groups.
        pdg_id: int
            The pdg_id of the particle.
        fourVector: FourVector
            The 4-vector for this particle, with units of c = 1.
    
    Methods:
        GetMomentumMagnitude:
            Gets the magnitude of the total momentum.
        AddFourVectors:
            Sums the four vectors of a list of ParticleEvents.
        GroupEventsByEventGroup:
            Groups a list of ParticleEvents together by their event groups.
        GetTransverseMomentum:
            Gets the magnitude of the momentum in the x-y plane.
        GetPsuedorapidity:
            Gets the angle from the z-axis (beam axis).
        GetPhi:
            Gets the angle in the x-y plane.
        GetRestMass:
            Gets the rest (invariant) mass.
        GetEnergy:
            Gets the energy.
        GetXMomentum:
            Gets the momentum in the x-direction.
        GetYMomentum:
            Gets the momentum in the y-direction.
        GetZMomentum:
            Gets the momentum in the z-direction.
    """

    def __init__(self, eventGroup: int, pdg_id: int, p_x: float, p_y: float, p_z: float, energy: float) -> None:
        self.eventGroup = eventGroup
        self.pdg_id = pdg_id
        self.fourVector = FourVector(energy, p_x, p_y, p_z)

    def __str__(self) -> str:
        return f"[eventGroup={self.eventGroup}, pdg_id={self.pdg_id}, fourVector={self.fourVector}]"

    def AddFourVectors(events: list[ParticleEvent]) -> FourVector:
        """Adds the four vectors of a list of particle events.

        Parameters:
            events: list[ParticleEvent]
                A list of particle events whose four vectors
                need to be summed.
        
        Returns:
            FourVector:
                The combined four vector.
        """
        
        # Creates a base four vector.
        combinedFourVector = FourVector(0, 0, 0, 0)
        
        # Adds all of the four vectors of the events to combine.
        for event in events:
            combinedFourVector += event.fourVector

        return combinedFourVector

    def GroupEventsByEventGroup(events: list[ParticleEvent]) -> list[list[ParticleEvent]]:
        """Groups a list of particle events by their event groups.

        The method assumes that the events are already in a list
        and any two events that have the same event group are
        next to each other in the list. This method simply takes this list
        and splits its up to more easily access these event groups.

        If the events are in random order, you CANNOT use this method.

        Ideally, this method should only be used within other methods, and
        should never be used as an input to a method (except in rare situations where
        the output only makes sense if the user already has the grouped list).
        Besides the rare exceptions, every function should take in a normal list
        of particle events as input.
        
        Parameters:
            events: list[ParticleEvent]
                The list of particles to group.

        Returns:
            list[list[ParticleEvent]]:
                A list, where each element in the list is
                a list of particle events with the same event group.
        """

        output = []

        # Stores the current event group.
        currentEventGroup = events[0].eventGroup
        # Stores all the events found so far within the current
        # event group.
        workingGroup = []
        for event in events:
            if (event.eventGroup == currentEventGroup):
                workingGroup.append(event)

            else:
                # Puts the list of all events in the current event group
                # at the end of output.
                output.append(workingGroup)

                # Resets the event group tracker.
                currentEventGroup = event.eventGroup
                # Resets the working group to just hold the
                # current event.
                workingGroup = [event]

        return output
    
    def CalculateDeltaR(event1: ParticleEvent, event2: ParticleEvent) -> float:
        """Calculates the Delta R between two events.
        
        Parameters:
            event1: ParticleEvent
                The first event to consider.
            event2: ParticleEvent
                The second event to consider.
                
        Returns:
            float:
                The delta R value between the two events.
        """
 
        deltaEta = event1.GetPsuedorapidity() - event2.GetPsuedorapidity()
        deltaPhi = event1.GetPhi() - event2.GetPhi()
        deltaR = math.sqrt(deltaEta**2 + deltaPhi**2)

        return deltaR


    def GetMomentumMagnitude(self) -> float:
        v = self.fourVector
        return math.sqrt(v.p_x * v.p_x + v.p_y * v.p_y + v.p_z * v.p_z)

    # Transverse momentum is the momentum in the x-y plane.
    def GetTransverseMomentum(self) -> float:
        v = self.fourVector
        return math.sqrt(v.p_x * v.p_x + v.p_y * v.p_y)
    
    # Psuedorapidity is the angle from the z-axis (beam axis).
    def GetPsuedorapidity(self) -> float:
        # Checks if the momentum is 0.
        # if (self.GetMomentumMagnitude() == 0):
        #     return 

        try:
            return math.atanh(self.fourVector.p_z / self.GetMomentumMagnitude()) # Uses the simplest formula for pseudorapidity.
        except:
            return 0

    
    # Phi is the angle of the detection in the x-y plane.
    def GetPhi(self) -> float:
        # Checks if the denominator is 0. If so, it'll return +/- pi/2.
        if (self.fourVector.p_x == 0):
            return math.pi * math.sign(self.fourVector.p_x) / 2

        return math.atan(self.fourVector.p_y / self.fourVector.p_x) # Uses the formula for phi given by Harish.
    
    def GetRestMass(self) -> float:
        return self.fourVector.Magnitude()

    def GetEnergy(self) -> float:
        return self.fourVector.energy

    def GetXMomentum(self) -> float:
        return self.fourVector.p_x
    
    def GetYMomentum(self) -> float:
        return self.fourVector.p_y
    
    def GetZMomentum(self) -> float:
        return self.fourVector.p_z