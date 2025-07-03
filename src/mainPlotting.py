# Default imports.

# Class imports for use as objects.

# Module imports for use of externally defined functions.
from dataparsing.DataIO import *
from dataparsing.Plotting import *

if __name__ == "__main__":
    # Loops through the SM and BSM data.
    for DATA_NAME in ("FourTopLHE_SM", "FourTopLHE_BSM"):
        DATA_FOLDER = "../data/" + DATA_NAME
        OUTPUT_FOLDER = "../output/" + DATA_NAME

        # Gets all the collisions.
        collisions = ExtractCollisionsFromFolder(DATA_FOLDER)
        print(f"Number of events: {len(collisions)}")
    
        # Filters just the top and anti-top particles.
        [collision.Filter([-6, 6]) for collision in collisions]
        print(f"Number of top events: {len(collisions)}")

        # Gets all the four vectors.
        fourVectors = [collision.GetFourVectors() for collision in collisions]
        print(f"Four Vectors Groups: {len(fourVectors)}")

        # Calculates all the possible delta Rs between the tops,
        # where the tops are ordered 0-3 in order of decreasing p_T.
        deltaRDictionaries = [collision.CalculateAllDeltaR() for collision in collisions]
        print(f"Delta R Groups: {len(deltaRDictionaries)}")

        # Calculates the invariant masses.
        invariantMasses = [collision.GetCombinedInvariantMass() for collision in collisions]
        print(f"Invariant Mass Groups: {len(invariantMasses)}")

        CSVData = []
        # Loops through each of the groups.
        for groupIndex in range(len(invariantMasses)):
            currentData = []

            # Loops through the four vectors.
            for vector in fourVectors[groupIndex]:
                currentData.append(vector.energy)
                currentData.append(vector.px)
                currentData.append(vector.py)
                currentData.append(vector.pz)

            # Loops through all the deltaRs in order and appends them.
            for firstParticle in range(0, 4):
                for secondParticle in range(firstParticle + 1, 4):
                    currentData.append(deltaRDictionaries[groupIndex][(firstParticle, secondParticle)])
    
            # Once the deltaRs have been added, adds the invariant mass.
            currentData.append(invariantMasses[groupIndex])

            # Appends this list to final data.
            CSVData.append(currentData)

        # Outputs the data to a CSV.
        header = "E-0,px-0,py-0,pz-0,E-1,px-1,py-1,pz-1,E-2,px-2,py-2,pz-2,E-3,px-3,py-3,pz-3,R-01,R-02,R-03,R-12,R-13,R-23,m0"

        PlotHistogram([x[0] for x in CSVData], fileName="E-0", folder=OUTPUT_FOLDER, xLabel="Energy (GeV)", yLabel="Frequency", fontSize=12)
        PlotHistogram([x[16] for x in CSVData], fileName="R-01", folder=OUTPUT_FOLDER, xLabel=r"$\Delta R$ (radians)", yLabel="Frequency", fontSize=12)
        PlotHistogram([x[22] for x in CSVData], fileName="m0", folder=OUTPUT_FOLDER, xLabel="Invariant Mass (GeV)", yLabel="Frequency", fontSize=12)