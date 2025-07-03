# Default imports.

# Class imports for use as objects.

# Module imports for use of externally defined functions.
from dataparsing.DataIO import *
from dataparsing.Plotting import *

if __name__ == "__main__":
    bothDatasets = []   

    # Loops through the SM and BSM data.    
    for DATA_NAME in ("FourTopLHE_SM", "FourTopLHE_BSM"):   
        DATA_FOLDER = "../data/" + DATA_NAME    
        OUTPUT_FOLDER = "../output/" + DATA_NAME    

        # Gets all the collisions.  
        collisions = ExtractCollisionsFromFolder(DATA_FOLDER)   
        print(f"Number of events: {len(collisions)}")   

        # Filters just the top and anti-top particles.  
        # [collision.Filter([-6, 6]) for collision in collisions]   
        # print(f"Number of top events: {len(collisions)}") 

        # Gets all the four vectors.    
        fourVectors = [collision.GetFourVectors() for collision in collisions]  
        print(f"Four Vectors Groups: {len(fourVectors)}")   

        # Calculates all the possible delta Rs between the tops,    
        # where the tops are ordered 0-3 in order of decreasing p_T.    
        # deltaRDictionaries = [collision.CalculateAllDeltaR() for collision in collisions] 
        # print(f"Delta R Groups: {len(deltaRDictionaries)}")   

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
            # for firstParticle in range(0, 4): 
            #     for secondParticle in range(firstParticle + 1, 4):    
            #         currentData.append(deltaRDictionaries[groupIndex][(firstParticle, secondParticle)])   
    
            # Once the deltaRs have been added, adds the invariant mass.    
            currentData.append(invariantMasses[groupIndex]) 

            # Appends this list to final data.  
            CSVData.append(currentData) 

        # Outputs the data to a CSV.    
        header = "E-0g,px-0g,py-0g,pz-0g,E-1g,px-1g,py-1g,pz-1g,E-0t,px-0t,py-0t,pz-0t,E-1t,px-1t,py-1t,pz-1t,E-2t,px-2t,py-2t,pz-2t,E-3t,px-3t,py-3t,pz-3t,m0" 
        OutputListToCSV(CSVData, OUTPUT_FOLDER + "/Tops & Leptons - Four-Vectors and m0.csv", header)   

        bothDatasets.append(CSVData)    

    # splitHeader = header.split(",")   
    # bins = 200    
    # # Loops through the index of each column being written to the CSV.    
    # for i in range(len(splitHeader)): 
    #     SMData = [x[i] for x in bothDatasets[0]]  
    #     BSMData = [x[i] for x in bothDatasets[1]] 
    #     # PlotHistogram(columnData, bins=200, title=f"{splitHeader[i]}", folder="output/SM-BSM Comparison")   
    #     plt.subplot(1, 2, 1)  
    #     plt.hist(SMData, bins=bins, label="SM")   
    #     plt.subplot(1, 2, 2)  
    #     plt.hist(BSMData, bins=bins, label="BSM") 
    #     plt.title(f"Comparison of {splitHeader[i]}")  
    #     plt.legend(loc='upper right') 
    #     plt.savefig(f"output/SM-BSM Comparison/{splitHeader[i]} Comparison")  
    #     plt.clf() 