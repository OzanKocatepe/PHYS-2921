from ParticleEvent import ParticleEvent

class CombinedParticleEvent(ParticleEvent):
    """Represents a particle event formed from combining multiple particle events from the same group.
    
    A combined particle event has an event group (since we're assuming the original particle events came from
    the same event group) and the pdg_id is 0 None. Then, its four vector is just the combined
    four vectors of its composite events.

    Attributes:
        subEvents: list[ParticleEvent]
            The list of events that form this combined event.
        numEvents: int
            The number of ParticleEvents combined to form this combined event.
    """
    def __init__(self, eventGroup: int, eventsToCombine: list[ParticleEvent]):
        self.subEvents = eventsToCombine
        combinedFourVector = ParticleEvent.AddFourVectors(eventsToCombine)

        # Uses the ParticleEvent constructor to create a ParticleEvent with pdg_id 0 and the combined four vector.
        super().__init__(eventGroup, 0, combinedFourVector.p_x, combinedFourVector.p_y, combinedFourVector.p_z, combinedFourVector.energy)
        
        # Sets or overrides the required variables that weren't set in the super constructor.
        self.pdg_id = None
        self.numEvents = len(self.subEvents)
