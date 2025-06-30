from ParticleEvent import ParticleEvent
import matplotlib.pyplot as plt

"""Contains functions related to plotting data.

Methods:
    PlotHistogram:
        Plots a histogram of the given data with the desired parameters.
    PlotEverything:
        Plots every possible attribute of a list of particle events.
"""

def PlotEverything(events: list[ParticleEvent], folder: str=".") -> None:
    """Plots all the attributes of a list of particle events onto their own histograms.
    
    Plots the x, y, z, and transverse momentums, energy, pseudorapidity, phi
    and the invariant mass on their own histograms. The titles, axes, and file names are
    automatically generated, and the histograms are saved to the given folder.
    
    Parameters:
        events: list[ParticleEvent]
            The list of particle events.
        folder: str
            The folder to save the histograms in. By default it is the current directory.
    """

    # Takes the individual attributes of each particle event in the list of filtered events.
    print("Extracting data...")
    xMomentum = [p.GetXMomentum() for p in events]
    yMomentum = [p.GetYMomentum() for p in events]
    zMomentum = [p.GetZMomentum() for p in events]
    transverseMomentum = [p.GetTransverseMomentum() for p in events]
    energy = [p.GetEnergy() for p in events]
    pseudorapidity = [p.GetPsuedorapidity() for p in events]
    phi = [p.GetPhi() for p in events]
    invariantMass = [p.GetRestMass() for p in events]

    print("Plotting...")
    bins=250
    PlotHistogram(xMomentum, bins=bins, title="X-Momentum", folder=folder, xLabel="Momentum (GeV)", yLabel="Frequency")
    PlotHistogram(yMomentum, bins=bins, title="Y-Momentum", folder=folder, xLabel="Momentum (GeV)", yLabel="Frequency")
    PlotHistogram(zMomentum, bins=bins, title="Z-Momentum", folder=folder, xLabel="Momentum (GeV)", yLabel="Frequency")
    PlotHistogram(transverseMomentum, bins=bins, title="Transverse Momentum", folder=folder, xLabel="Momentum (GeV)", yLabel="Frequency")
    PlotHistogram(energy, bins=bins, title="Energy", folder=folder, xLabel="Energy (GeV)", yLabel="Frequency")
    PlotHistogram(pseudorapidity, bins=bins, title="Pseudorapidity", folder=folder, xLabel="Psuedorapidity (unitless)", yLabel="Frequency")
    PlotHistogram(phi, bins=bins, title="Phi", folder=folder, xLabel="Angle (degrees)", yLabel="Frequency")
    PlotHistogram(invariantMass, bins=bins, title="Invariant Mass", folder=folder, xLabel="Mass (GeV)", yLabel="Frequency")

def PlotHistogram(data: list, bins: int=25, title: str=None, folder: str=".", xLabel: str=None, yLabel: str=None) -> None:
    """Plots a histogram with the desired parameters and saves it to the given file.

    Parameters:
        data: list
            A list containing the values to plot.
        bins: int
            The width of the bins. By default it is 25.
        title: str
            The title of the histogram and the file name it is saved as.
        folder: str
            The folder to save the histogram in. By default it is the current directory.
        xLabel: str
            The label on the x-axis.
        yLabel: str
            The label on the y-axis.
    """

    # Creates the histogram.
    plt.hist(data, bins=bins)

    # Applies labels if they are given.
    if title != None:
        plt.title(title)
    if xLabel != None:
        plt.xlabel(xLabel)
    if yLabel != None:
        plt.ylabel(yLabel)

    # Saves the figure and then clears the plot.
    plt.savefig(folder + "/" + title)
    plt.clf()