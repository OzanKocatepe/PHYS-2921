import sys
from dataparsing import Manipulation, DataIO

# Checks if this is __main__ so that sphinx doesn't run the code
# when generating docs.
if __name__ == "__main__":

    # Initialises the flags.
    inputFolder = None
    outputFile = None
    includeFourVectors = False
    includeDeltaR = False
    includeInvariantMass = False

    # Looks for the command flags.
    for i in range(1, len(sys.argv)):

        # Outputs the help page from the file that it is stored in.
        if (sys.argv[i] == "help"):
            
            file = open("HelpMessage.txt", 'r')
            helpLines = file.readlines()
            file.close()

            print("")
            for line in helpLines:
                print(line, end="")
            print("\n")
            exit()

        elif (sys.argv[i] == "-o"):
            # Gets the next flag as the output file name.
            outputFile = sys.argv[i + 1]

        elif (sys.argv[i] == "-v"):
            includeFourVectors = True

        elif (sys.argv[i] == "-r"):
            includeDeltaR = True

        elif (sys.argv[i] == "-m"):
            includeInvariantMass = True

        # If its not a flag, and its not the output file name, then its the input folder.
        else:
            if (sys.argv[i - 1] != "-o"):
                inputFolder = sys.argv[i]

    # Handles errors.
    if (inputFolder == None):
        raise Exception("No input file specified.")
    if (outputFile == None):
        raise Exception("No output file specified.")
    
    # Extracts the collision data from the input folder.
    collisions = DataIO.ExtractCollisionsFromFolder(inputFolder)
    [collision.Filter([6, -6]) for collision in collisions]

    # Gets the desired data and outputs it to the CSV.
    csvData = Manipulation.ExtractAttributes(collisions, includeFourVectors, includeDeltaR, includeInvariantMass)
    DataIO.OutputDictToCSV(csvData, outputFile)