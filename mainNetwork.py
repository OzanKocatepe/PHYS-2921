from NeuralNetwork import NeuralNetwork

NODES = (128, 32, 64, 192)             # How many nodes to have in each hidden layer.
EPOCHS = 100                           # Number of epochs to train for.
BATCH_SIZE = 100                       # Size of each batch sent to the neural network at once.
LEARNING_RATE = 0.00043                # How fast to do the gradient descent, i.e. how fast to change the weights.
VALIDATION_SPLIT = 0.2                 # Percentage of data to use for validation.
TESTING_SPLIT = 0.2                    # Percentage of data to use for testing.

# Input files.
CSV_FILE_NAME = "Tops - Four-Vectors, DeltaR and m0" 
SM_CSV = f"output/FourTopLHE_SM/{CSV_FILE_NAME}.csv"
BSM_CSV = f"output/FourTopLHE_BSM/{CSV_FILE_NAME}.csv"

network = NeuralNetwork(VALIDATION_SPLIT, TESTING_SPLIT)
network.InputData(SM_CSV, BSM_CSV)
network.TuneHyperParameters()