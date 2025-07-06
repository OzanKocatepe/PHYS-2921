from objects.Collision import Collision

"""Contains the functions for manipulating the gathered data."""

def ExtractAttributes(collisions: list[Collision],
                      includeFourVectors: bool=False,
                      includeDeltaR: bool=False,
                      includeInvariantMass: bool=False) -> list[dict]:
    """Extracts all the desired attributes from a list of collisions.
    
    Parameters:
        collisions (list[Particle]):
            The list of collisions to extract the data from.
        includeFourVectors (bool):
            Whether to include the four vectors of the particles.
        includeDeltaR (bool):
            Whether to include the delta R separations of the particles.
        includeInvariantMass (bool):
            Whether to include the combined invariant mass of each collision.
    
    Returns:
        list[dict]:
            A list of dictionaries, with each dictionary containing the desired attributes
            extracted from a single collision.
    """

    output = []

    for collision in collisions:
        dict = {}

        if (includeFourVectors):
            fourVectors = collision.GetFourVectors()
            # Converts the array of 4-vectors into a list of floats, where
            # each 4-vector is stored successively.
            fourVectorsList = []

            for vec in fourVectors:
                fourVectorsList += vec.toList()

            dict["fourVectors"] = fourVectorsList

        if (includeDeltaR):
            deltaR = collision.CalculateAllDeltaR()
            
            dict["deltaR"] = deltaR

        if (includeInvariantMass):
            dict["invariantMass"] = collision.GetCombinedInvariantMass()
        