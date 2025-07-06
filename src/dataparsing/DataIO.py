import os

from objects.Collision import Collision
from objects.Particle import Particle

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

def ReadListFromCSV(fileName: str) -> tuple[list[str], list[tuple[float]]]:
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

def ReadDictFromCSV(fileName: str) -> list[dict]:
    """Reads the data from a CSV as a list of dictionaries.
    
    Parameters:
        fileName (str):
            The path to the CSV file.

    Returns:
        list[dict]:
            A list of dictionaries, where each dictionary contains the information
            from a single line. The keys of each dictionary are the corresponding header the
            data was stored under. If there was no data stored, a None type is given instead.
    """

    data = []

    file = open(fileName, 'r')
    lines = file.readlines()
    file.close()

    # Reads the keys from the header.
    keys = lines[0].split("\n")
    keys = keys.split(",")

    # Loops through all lines after the header.
    for line in lines[1:]:
        dict = {}

        values = line.split("\n")
        values = values.split(",")

        # Loops through all the given values.
        for i in range(len(values)):
            dict[keys[i]] = values[i]

        # Any values not given are set to be None.
        # If all values (or extra) values are given, this code
        # won't run.
        for i in range(len(values), len(keys)):
            dict[keys[i]] = None

        data.append(dict)

    return data

def OutputDictToCSV(data: list[dict], fileName: str) -> None:
    """Outputs a list of dictionaries to a CSV.
    
    Parameters:
        data (list[dict]):
            A list of dictionaries, where each dictionary contains the same set
            of keys. Each dictionary will be written to one line.
        fileName (str):
            The file path to write the CSV to.
    """

    file = open(fileName, 'w')

    keys = list(data[0].keys())

    # Writes the keys to the header of the file.
    for key in keys[0:-1]:
        file.write(f"{key},")
    file.write(f"{keys[-1]}\n")
    
    for line in data:
        for key in keys[0:-1]:
            file.write(f"{line[key]},")
        file.write(f"{line[keys[-1]]}\n")
            