import matplotlib.pyplot as plt

"""Contains functions related to plotting data."""

def PlotHistogram(data: list, fileName: str, bins: int=200, folder: str=".", title: str=None, xLabel: str=None, yLabel: str=None, fontSize: int=10) -> None:
    """Plots a histogram with the desired parameters and saves it to the given file.

    Parameters:
        data (list):
            A list containing the values to plot.
        fileName (str):
            The name to give the saved histogram file.
        bins (int):
            The width of the bins. 
        folder (str):
            The folder to save the histogram in. By default it is the current directory.
        title (str):
            The title of the histogram.
        xLabel (str):
            The label on the x-axis.
        yLabel (str):
            The label on the y-axis.
        fontSize (int):
            The font size of the title and axis labels.
    """

    # Creates the histogram.
    plt.hist(data, bins=bins)

    # Applies labels if they are given.
    if title != None:
        plt.title(title, fontsize=fontSize)
    if xLabel != None:
        plt.xlabel(xLabel, fontsize=fontSize)
    if yLabel != None:
        plt.ylabel(yLabel, fontsize=fontSize)

    # Saves the figure and then clears the plot.
    plt.savefig(folder + "/" + fileName)
    plt.clf()