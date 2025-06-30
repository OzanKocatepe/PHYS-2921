import os
import math

from ParticleEvent import ParticleEvent
from EventCounter import EventCounter
from FourVector import FourVector
from CombinedParticleEvent import CombinedParticleEvent

"""Contains the functions related to parsing the required data from files.

The functions are used for parsing events out of the event files and
filtering/analysing this data.

Methods:
    ExtractEventsFromFile:
        Extracts every event from a file of events.
    ExtractEventsFromFolder:
        Extracts every event from a folder full of only event files.
    FindParticleInstances:
        Filters a list of ParticleEvents to only contain desired particles.
    FindDeltaRForPairs:
        Finds the Delta R seperation for single pairs of particles in an event group.
    CombineEventGroupsIntoCombinedEvents:
        Combines a list of ParticleEvents into CombinedParticleEvents by event group.
    OutputEventsToCSV:
        Outputs a list of events to a .csv file.
    CalculateCombinedFourVectors:
        Calculates the combined four vectors for a list of
        event groups.
    OrderGroup:
        Orders a group of particle events by decreasing
        transverse momentum.
    CalculateAllDeltaR:
        Calculates the Delta R for every possible pair of
        particles in each group in a list of particle events.
    CalculateAllCombinedInvariantMasses:
        Calculates the combined invariant masses for each group
        in a list of particle events.
    OutputListToCSV:
        Outputs a list of lists to a CSV.
"""

def ExtractEventsFromFile(fileName: str) -> list[ParticleEvent]:
    """Extracts all the events from a data file.
    Given a file of events, parses each line in each event and finds
    stores each individual event as a ParticleEvent instance.
    Parameters:
        fileName: str
            The name of the file to read the data from.
    Returns:
        list[ParticleEvent]:
            A list of the found particle events.
    """

    file = open(fileName, "r")
    line = file.readline()
    events = []
    # Loops through the file line-by-line until the file ends.
    while line:
        if (line == "<event>\n"):
            # Gets the event group (i.e. tracks which event we are reading).
            eventGroup = EventCounter.currentEvent
            # Reads the first line of the event.
            line = file.readline()
            # Gets the number of particles in the event.
            numParticles = int((line.split())[0])
            # Loops through each particle event.
            for i in range(numParticles):
                line = file.readline()
                line = line.split()
                # print(line)
                # Grabs the relevant information from the particle event.
                pdg_id = int(line[0])
                p_x = float(line[6])
                p_y = float(line[7])
                p_z = float(line[8])
                energy = float(line[9])
                # restMass = float(line[10])
                # Adds the event to the events list as a ParticleEvent object.
                events.append(ParticleEvent(eventGroup, pdg_id, p_x, p_y, p_z, energy))

            # Increments the event counter now that we have read all detections in the event.
            EventCounter.IncrementCounter()
        
        # Gets next line from file.
        line = file.readline()
    return events

def ExtractEventsFromFolder(folder: str, debug: bool=False) -> list[ParticleEvent]:
    """Extracts particle events from a folder full of event files.
    Given a folder of event files, this parses the events from each of them
    and returns the final list. Assumes all of the files in the folder are event files
    of the appropriate format.
    Parameters:
        folder: str
            The name of the folder containing the event files.
        debug: bool
            Whether to print debug statements during execution. False by default.
    Returns:
        events:
            The list of particle events corresponding to the desired particles
            in the event files.
    """

    events = []
    for file in os.listdir(folder):
        if debug:
            print(f"Currently parsing: {file}")
        
        events += ExtractEventsFromFile(folder + "/" + file)
    return events

def FindParticleInstances(particles: list[int], events: list[ParticleEvent]) -> list[ParticleEvent]:
    """Finds the ParticleEvents corresponding to specific particles.
    Given a list of ParticleEvents, it returns a new list containing only
    the ParticleEvents corresponding to specific particles.
    Parameters:
        particles: list[int]
            The pdg_ids of the desired particles.
        events: list[ParticleEvent]
            The list of ParticleEvents to look through.
    Returns:
        list[ParticleEvent]:
            A list containing only the ParticleEvents of the desired particles.
    """
    # Creates a list to store the desired events in.
    filteredEvents = []
    # Loops through all the particle events in events,
    # finding the particle events for the desired particle.
    for p in events:
        if (p.pdg_id in particles):
            filteredEvents.append(p)
    return filteredEvents

def FindDeltaRForPairs(events: list[ParticleEvent]) -> list[float]:
    """Calculates the Delta R for the first pair of particles in each event.
    
    Parameters:
        events:
            The list of ParticleEvents to look through.
        
    Returns:
        list[float]:
            A list of the Delta R seperations.
    """

    deltaRList = []

    groups = ParticleEvent.GroupEventsByEventGroup(events)
    for group in groups:
        if (len(group) >= 2):
            deltaRList.append(ParticleEvent.CalculateDeltaR(group[0], group[1]))

    return deltaRList

def CombineEventGroupsIntoCombinedEvents(events: list[ParticleEvent]) -> list[CombinedParticleEvent]:
    """Combines a list of ParticleEvents into CombinedParticleEvents by event group.
    
    Parameters:
        events: list[ParticleEvent]
            The list of events to combine by event group.
            Assumes the list is NOT already sorted by group.
        
    Returns:
        list[CombinedParticleEvent]:
            The list containing the combined particle events.
    """

    # Stores the combined particle events.
    combinedEvents = ParticleEvent.GroupEventsByEventGroup(events)
    combinedEvents = [CombinedParticleEvent(collection[0].eventGroup, collection) for collection in combinedEvents]

    return combinedEvents

def OutputEventsToCSV(events: list[ParticleEvent], fileName: str, maxEvents: int=4) -> None:
    """Outputs a set of events to a .csv file.

    The .csv is formatted such that each line contains the four vectors of each particle
    in one event group, followed by their combined invariant mass (four-vectors are combined
    first, and then the invariant mass is found).

    Will aim to write four events to the file per line. If there aren't 4,
    it will fill in the rest of the events with (-1, -1, -1, -1) four vectors.

    Parameters:
        groupedEvents: [list[ParticleEvent]
            A list of particle events to output to the .csv file.
        fileName: str
            The desired file path for the .csv file. The program assumes that
            the folder that the file will be created in already exists.
            If the file itself already exists, the output of this
            function will overwrite the content in the file.
        maxEvents: int
            The desired number of events in each line. i.e., if there are less than this number,
            fill the rest of the line with (-1, -1, -1, -1) vectors.
    """

    # Opens the file in write mode, so anything in the file will be overwritten.
    file = open(fileName, "w")

    # Groups the events by event group.
    groupedEvents = ParticleEvent.GroupEventsByEventGroup(events)

    for i in range(maxEvents):
        file.write("E,px,py,pz,")
    file.write("m0\n")

    # Iterates through each event group, so listOfevents is a list.
    for listOfEvents in groupedEvents:
        # Keeps track of how many events are in this group.
        numEvents = 0

        # Iterates through the individual detections in the event group.
        for event in listOfEvents:
            # Writes the four vectors to the file.
            v = event.fourVector
            file.write(f"{v.energy},{v.p_x},{v.p_y},{v.p_z},")

            # Increments the counter.
            numEvents += 1

        # Calculates the combined invariant mass.
        combinedFourVector = ParticleEvent.AddFourVectors(listOfEvents)
        invariantMass = combinedFourVector.Magnitude()

        for i in range(numEvents, maxEvents):
            file.write(f"-1,-1,-1,-1,")
        file.write(f"{invariantMass}\n")

    file.close()

def CalculateCombinedFourVectors(groups: list[list[ParticleEvent]]) -> list[FourVector]:
    """Calculates the combined four vectors for a list of event groups.
    
    Parameters:
        groups: list[list[ParticleEvent]]
            A list of lists, where each element is a list of
            particle events which are all in the same event group.
            In the form outputted by ParticleEvent.GroupEventsByEventGroup
    
    Returns:
        list[FourVector]:
            A list of the combined four vectors, where the i-th
            index corresponds to the group at the i-th position in
            groups.
    """

    fourVectors = []

    # Loops through all the groups.
    for group in groups:
        # Appends the combined four vector to the output list.
        fourVectors.append(ParticleEvent.AddFourVectors(group))

    return fourVectors

def OrderGroup(events: list[ParticleEvent]) -> list[ParticleEvent]:
    """Given a list of particle events, orders them in order of decreasing transverse momentum.

    NOTE: Since events is a list, it is mutable, so this may change the value of
    events in place. We assume that if we are calling this function, we are aiming to replace
    the current value of events with the ordered value anyways.
    Furthermore, we hope that there is never a reason that the list *shouldn't* be ordered.
    However, this is an important place to keep an eye out, it could cause an unexpected bug.

    Parameters:
        events: list[ParticleEvent]
            The list of particle events to parse through.
    
    Returns:
        list[ParticleEvent]:
            The same list of particle events ordered in order of decreasing tranverse momentum.
    """

    sorted = False
    # While the list is not sorted, continues looping.
    while (not sorted):
        # Sets sorted to true. If we go through the entire list without
        # swapping anything, then the list is sorted.
        sorted = True

        # Loops through the entire list, starting at index 1.
        for i in range(1, len(events)):
            # Compares current element to the one before it.
            # Swaps if they're out of order.
            if (events[i - 1].GetTransverseMomentum() < events[i].GetTransverseMomentum()):
                events[i - 1], events[i] = events[i], events[i - 1]
                # Notes that we needed a swap, so the list is not certainly sorted.
                sorted = False

    return events

def CalculateAllDeltaR(events: list[ParticleEvent]) -> list[dict[float]]:
    """Calculates all the possible delta R's for a set of particle events.
    
    Parameters:
        events: list[ParticleEvent]:
            The list of ParticleEvents to parse through.

    Returns:
        list[dict[float]]:
            A list of dictionaries, where each dictionary contains
            the delta R value for an ordered group of particles.
            The keys are of the form (i, j), where
            1 <= i < j <= number of particles in current group,
            and i and j are the indexes in the ordered grouped list
            of particle events (previous returned value) of the
            two particles being compared.
    """

    groupedDeltaRs = []

    groups = ParticleEvent.GroupEventsByEventGroup(events)
    # Iterates through the grouped list.
    for group in groups:
        # Orders the group in order of descending transverse momentum.
        group = OrderGroup(group)

        # Defines a dictionary to save the delta R values to.
        deltaRs = {}

        # Loops through every possible pair.
        for firstIndex in range(0, len(group)):
            for secondIndex in range(firstIndex + 1, len(group)):
                # Saves the delta R to the dictionary
                # under the key (firstIndex, secondIndex)
                deltaRs[(firstIndex, secondIndex)] = ParticleEvent.CalculateDeltaR(group[firstIndex], group[secondIndex])

        groupedDeltaRs.append(deltaRs)
    
    return groupedDeltaRs

def CalculateAllCombinedInvariantMasses(events: list[ParticleEvent]) -> list[float]:
    """Calculates the combined invariant mass of each group of particles.
    
    Parameters:
        events (list[ParticleEvent]):
            The list of particle events to look through.
    
    Returns:
        list[float]:
            A list of the invariant masses for each group
            in events. The order of the invariant masses
            matches the order in which each group appears
            in events.
    """

    combinedInvariantMasses = []

    # Groups the events by event group.
    groups = ParticleEvent.GroupEventsByEventGroup(events)
    # Calculates the combined four vectors.
    combinedFourVectors = CalculateCombinedFourVectors(groups)
    # Calculates the invariant masses.
    combinedInvariantMasses = [vector.Magnitude() for vector in combinedFourVectors]

    return combinedInvariantMasses

def OutputListToCSV(data: list[list], fileName: str, header: str) -> None:
    """Outputs a list of lists to a CSV.
    
    Each entry in the list constitutes one line.
    Will overwrite any content already in the file.
    
    Parameters:
        data (list[list]):
            A list of lists that contains the data you desire on each line.
        fileName (str):
            The file to save the data to.
        header (str):
            The header to write at the top of the file.
    """

    file = open(fileName, 'w')
    file.write(f"{header}\n")
    for line in data:
        for i in range(len(line) - 1):
            file.write(f"{line[i]},")
        file.write(f"{line[i + 1]}\n")

def GetFourVectorsGrouped(events: list[ParticleEvent]) -> list[list[FourVector]]:
    """Gets all of the four vectors for each group.
    
    Parameters:
        events (list[ParticleEvent]):
            The list of particle events.

    Returns:
        list[list[FourVector]]:
            The first dimension correponds to the group.
            The second dimension corresponds to the particle,
            arranged by order of decreasing transverse momentum.
            Each final element is a four vector.
    """

    groupedFourVectors = []

    # Groups the events.
    groups = ParticleEvent.GroupEventsByEventGroup(events)

    # Loops through the groups.
    for group in groups:
        # First orders the group.
        orderedGroup = OrderGroup(group)

        fourVectors = []
        # Gets all the four vectors of the particles in the ordered group.
        for particle in orderedGroup:
            fourVectors.append(particle.fourVector)

        groupedFourVectors.append(fourVectors)

    return groupedFourVectors