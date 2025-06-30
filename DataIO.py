import os

from Collision import Collision
from Particle import Particle

"""Contains the functions used to extract data from a file."""

def ExtractCollisionsFromFile(fileName: str) -> list[Collision]:
    """Extracts all of the collisions from a file.

    Parameters:
        fileName (str):
            The name of the file to read the data from.

    Returns:
        list[Collision]:
            A list of the found particle events.
    """

    # Opens the file and reads all the lines.
    file = open(fileName, "r")

    collisions = []
    # Loops through the file line-by-line.
    while (line := file.readline()):
        
        # If there is a starting event tag, we begin to parse the data from the event.
        if (line == "<event>\n"):
            # Reads the first line of the event.
            line = file.readline()

            # Gets the number of particles in the event.
            numParticles = int((line.split())[0])

            # Creates a list to store the particle detections in.
            particles = []

            # Loops through each particle in the event.
            for i in range(numParticles):
                line = file.readline()
                line = line.split()
                
                # Grabs the relevant information from the particle event.
                pdg_id = int(line[0])
                px = float(line[6])
                py = float(line[7])
                pz = float(line[8])
                energy = float(line[9])
    
                # Adds the event to the events list as a ParticleEvent object.
                particles.append(Particle(pdg_id, energy, px, py, pz))

            # Once we have found all the particles in the event, creates a new
            # collision object and appends it to the list of collisions.
            collisions.append(Collision(particles))

    file.close()
    return collisions

def ExtractCollisionsFromFolder(folderName: str, debug: bool=True) -> list[Collision]:
    """Extracts all of the collisions from a folder of files.
    
    NOTE: Assumes all the file are in a valid format. If there are other
    types of files in the folder, this function will break.

    Parameters:
        folderName (str):
            The name of the folder to search through.
        debug (bool):
            Whether to print out progress updates.
    
    Returns:
        list[Collision]:
            A list of all the collisions found in the folder.
    """

    collisions = []
    
    # Loops through each file in the folder.
    for file in os.listdir(folderName):
        fileName = f"{folderName}/{file}"

        # Prints out the folder names.
        if debug:
            print(f"Currently parsing {fileName}...")

        # Extracts all of the collisions from the file and appends
        # them to the list of collisions already obtained.
        collisions += ExtractCollisionsFromFile(fileName)

    return collisions

def OutputListToCSV(data: list[list], fileName: str, header: str) -> None:
    """Outputs a list of lists to a CSV.
    
    Each entry in the list constitutes one line.
    Will overwrite any content already in the file.
    
    Parameters:
        data (list[list]):
            A list of lists that contains the data desired on each line.
        fileName (str):
            The file to save the data to. Will overwrite any data already
            saved in this file.
        header (str):
            The header to write at the top of the file.
    """
    
    # Opens the file.
    file = open(fileName, 'w')
    # Writes the header in.
    file.write(f"{header}\n")

    # Loops through each line.
    for line in data:
        # Loops through the data within the current line.
        for i in range(len(line) - 1):
            # Writes each entry of the line one by one.
            file.write(f"{line[i]},")
        # Writes the last entry with a newline instead of a comma.
        file.write(f"{line[i + 1]}\n")

    file.close()

def ReadDataFromCSV(fileName: str) -> tuple[list[str], list[tuple[float]]]:
    """Reads data from a CSV file.
    
    Parameters:
        fileName (str):
            The file to read the data from.
            
    Returns:
        list[string]:
            The header of the CSV.
        list[tuple[float]]:
            A list of tuples, where each tuple contains all of
            the data from a single line in the file.
    """

    data = []

    # Opens the file.
    file = open(fileName, 'r')
    # Gets all the lines of a file as a list.
    lines  = file.readlines()

    # Gets the first line of the file.
    header = lines[0].split("\n")
    header = header[0].split(",")

    # Loops through all the lines except the first one.
    for line in lines[1:]:
        # Converts all the data in the line into floats.
        # Appends the data as a tuple.
        splitLine = [float(x) for x in line.split(",")]
        data.append(tuple(splitLine))

    file.close()
    return header, data