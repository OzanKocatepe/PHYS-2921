import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.saving import load_model
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam

from keras_tuner import HyperParameters, RandomSearch

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_curve, auc

from dataparsing.Plotting import PlotHistogram
from dataparsing.DataIO import ReadDataFromCSV

class NeuralNetwork:
    """Neural Network for differentiating between SM and BSM particle events.
    
    Attributes:
        model (Sequential):
            The network itself.
        inputDataNames (list[str]):
            The names of each field of the input data.
        nodes (tuple[int]):
            A tuple that contains the number of nodes in each layer, excluding the implied
            final layer which always has one node.
        epochs (int):
            The number of epochs to train for.
        batchSize (int):
            The number of samples to send to the network at once.
        learningRate (float):
            How aggressively to change the weights each iteration.
        validationSplit (float):
            The percentage of the data to use for validation.
        testingSplit (float):
            The percentage of the data to use for testing.
        outputFolder (str):
            The root output folder that everything will be stored in.
        histogramOutputFolder (str):
            The subfolder that the input data histograms will be saved in.
        historyOutputFolder (str):
            The subfolder that the history data will be saved in.
        predictionsOutputFolder (str):
            The subfolder that the prediction histograms will be saved in.
        tuningOutputFolder (str):
            The subfolder that the hyperparameter tuning outputs will be saved in.
        trainingSamples (list[float]):
            The samples to use for training.
        trainingLabels (list[int]):
            The labels for the training samples.
        validationSamples (list[float]):
            The samples to use for validation.
        validationLabels (list[float]):
            The labels for the validation samples.
        testingSamples (list[float]):
            The samples to use for testing.
        testingLabels (list[float]):
            The labels for the testing samples.
        history (History):
            The history object obtained when the model is trained.
        trainingPredictions (list[float]):
            The predictions made by the trained neural network on the training data.
        validationPredictions (list[float]):
            The predictions made by the trained neural network on the validation data.
        testingPredictions (list[float]):
            The predictions made by the trained neural network on the testing data.
    """

    def __init__(self, validationSplit: float, testingSplit: float, rootOutputFolder: str="models/SM vs BSM") -> None:
        """Creates an instance of the NeuralNetwork.
        
        Parameters:
            validationSplit (float):
                The percentage of the data to use for validation.
            testingSplit (float):
                The percentage of the data to use for testing.
            rootOutputFolder (str):
                The root output folder where all outputs and subfolders will be saved.
        """

        self.validationSplit = validationSplit
        self.testingSplit = testingSplit

        self._CreateRequiredFolders(rootOutputFolder)

    def _CreateRequiredFolders(self, rootOutputFolder: str) -> None:
        """Creates all of the required folders and subfolders to save the model data to.

        Parameters:
            rootOutputFolder (str):
                The path where the root output folder should be created.
        """
        # Determines what the next valid folder is.
        outputFolderIndex = 0
        while os.path.exists(outputFolder := f"{rootOutputFolder}/model-{outputFolderIndex}"):
            outputFolderIndex += 1

        # Stores the folder paths.
        self.outputFolder = outputFolder
        self.histogramOutputFolder = f"{self.outputFolder}/Input Data Plots"
        self.historyOutputFolder = f"{self.outputFolder}/History"
        self.predictionsOutputFolder = f"{self.outputFolder}/Predictions"

        # Creates all of the required directories.
        os.mkdir(self.outputFolder)
        os.mkdir(self.histogramOutputFolder)
        os.mkdir(self.historyOutputFolder)
        os.mkdir(self.predictionsOutputFolder)
    
    def InputData(self, smCSV: str, bsmCSV: str, bins: int=200) -> None:
        """Reads the input data from two CSVs, formats it, and plots it.
        
        Parameters:
            smCSV (str):
                The path to the CSV containing the SM data.
            bsmCSV (str):
                The path to the CSV containing the BSM data.
            bins (int):
                The number of bins to use when plotting the input data.
        """

        # Reads the information from the CSVs.
        header, dataSM = ReadDataFromCSV(smCSV)
        header, dataBSM = ReadDataFromCSV(bsmCSV)

        # Stores the names of the data.
        self.inputDataNames = header

        fullDatasetSamples = dataSM + dataBSM
        fullDatasetLabels = [0 for i in dataSM] + [1 for i in dataBSM]

        # Converts to numpy arrays.
        fullDatasetSamples = np.array(fullDatasetSamples)
        fullDatasetLabels = np.array(fullDatasetLabels)
        # Shuffles the data up.
        fullDatasetSamples, fullDatasetLabels = shuffle(fullDatasetSamples, fullDatasetLabels)

        # Plots the data before normalisation.
        for i in range(len(self.inputDataNames)):
            PlotHistogram([x[i] for x in fullDatasetSamples],
                          fileName=f"Pre-Normalised {self.inputDataNames[i]} Distribution",
                          bins=bins,
                          title=f"Pre-Normalised {self.inputDataNames[i]} Distribution",
                          folder=self.histogramOutputFolder)

        # Scales each attribute of the sample data from 0-1.
        scaler = MinMaxScaler()
        fullDatasetSamples = scaler.fit_transform(fullDatasetSamples)

        # Plots the data after normalisation.
        for i in range(len(self.inputDataNames)):
            PlotHistogram([x[i] for x in fullDatasetSamples],
                          fileName=f"Normalised {self.inputDataNames[i]} Distribution",
                          bins=bins,
                          title=f"Normalised {self.inputDataNames[i]} Distribution",
                          folder=self.histogramOutputFolder)

        # Splits the data into training, validation, and testing.
        numSamples = len(fullDatasetSamples)
        self.validationSamples, self.testingSamples, self.trainingSamples = np.split(fullDatasetSamples, [int(numSamples * self.validationSplit), int(numSamples * (self.validationSplit + self.testingSplit))])
        self.validationLabels, self.testingLabels, self.trainingLabels = np.split(fullDatasetLabels, [int(numSamples * self.validationSplit), int(numSamples * (self.validationSplit + self.testingSplit))])

    def CreateModel(self, nodes: tuple[int]) -> None:
        """Creates the actual model.
        
        Parameters:
            nodes (tuple[int]):
                A tuple indicating how many nodes each hidden layer should have.
                The final layer should not be included - it is implicit that the output
                layer will always have one node.
        """

        # Saves the node information.
        self.nodes = nodes

        # Creates the first layer, specifying input shape.
        self.model = Sequential(Dense(units=nodes[0], input_shape=(len(self.trainingSamples[0])), activation='relu'))
        # layers = [Dense(units=nodes[0], input_shape=(len(self.trainingSamples[0]),), activation='relu')]

        # Automatically creates the rest of the input layers based on the number of nodes.
        # All of them use the relu activation function.
        for i in range(1, len(nodes)):
            self.model.add(Dense(units=nodes[i], activation='relu'))

        # Defines the output layer to have 1 unit and a sigmoid activation function.
        self.model.add.append(Dense(units=1, activation='sigmoid'))

        # Creates the model.
        # self.model = Sequential(layers)

        # Prints summary of the model.
        self.model.summary()

    def TrainModel(self, epochs: int, batchSize: int, learningRate: float) -> None:
        """Trains the model using the input data.
        
        Parameters:
            epochs (int):
                How many epochs to train the model for.
            batchSize (int):
                How many samples to train the network on at once.
            learningRate (float):
                How aggresively to change the weights in each iteration.
        """

        # Stores the model information.
        self.epochs = epochs
        self.batchSize = batchSize
        self.learningRate = learningRate

        # Compiles the model.
        self.model.compile(optimizer=Adam(learning_rate=learningRate), loss='binary_crossentropy', metrics=['accuracy'])

        # Trains the model with the training data and gives it the validation data to validate itself on.
        self.history = self.model.fit(self.trainingSamples,
                                      self.trainingLabels,
                                      validation_data=(self.validationSamples, self.validationLabels),
                                      batch_size=self.batchSize,
                                      epochs=self.epochs,
                                      verbose=2)

    def SaveModel(self) -> None:
        """Saves the model and the model's information to the output folder."""

        # Saves the model.
        self.model.save(f"{self.outputFolder}/model.keras")

        # Saves all of the model information.
        file = open(f"{self.outputFolder}/Model Information.txt", 'w')
        file.write(f"Input data: {self.inputDataNames}\n")
        file.write(f"Layers: {self.nodes}\n")
        file.write(f"Epochs: {self.epochs}\n")
        file.write(f"Batch size: {self.batchSize}\n")
        file.write(f"Learning rate: {self.learningRate}\n")
        file.write(f"Validation Split: {self.validationSplit}\n")
        file.write(f"Testing Split: {self.testingSplit}\n\n")

        file.write(f"Final Training Accuracy = {self.history.history['accuracy'][-1]}\n")
        file.write(f"Final Validation Accuracy = {self.history.history['val_accuracy'][-1]}\n")
        file.write(f"Final Training Loss = {self.history.history['loss'][-1]}\n")
        file.write(f"Final Validation Loss = {self.history.history['val_loss'][-1]}")

        file.close()

    def PlotAccuracyAndLoss(self) -> None:
        """Plots the models accuracy and loss on the training and validation data over time,
        and saves the plots to the output folder.
        """

        # Saves the training history.
        np.save(f"{self.historyOutputFolder}/history.npy", self.history)

        # Gets the accuracy and loss values over time during training.
        trainingAccuracy = self.history.history['accuracy']
        validationAccuracy = self.history.history['val_accuracy']
        trainingLoss = self.history.history['loss']
        validationLoss = self.history.history['val_loss']
        epochsRange = range(1, len(trainingAccuracy) + 1)

        # Plots the accuracy over time.
        plt.plot(epochsRange, trainingAccuracy, label="Training Accuracy")
        plt.plot(epochsRange, validationAccuracy, label="Validation Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(f"{self.historyOutputFolder}/Accuracy over Time.png")
        plt.clf()

        # Plots the loss over time.
        plt.plot(epochsRange, trainingLoss, label="Training Loss")
        plt.plot(epochsRange, validationLoss, label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{self.historyOutputFolder}/Loss over Time.png")
        plt.clf()

    def MakePredictions(self) -> None:
        # Makes predictions on each set of data.
        self.trainingPredictions = self.model.predict(
            self.trainingSamples,
            batch_size=self.batchSize,
            verbose=0
        )
        
        self.validationPredictions = self.model.predict(
            self.validationSamples,
            batch_size=self.batchSize,
            verbose=0
        )
        
        self.testingPredictions = self.model.predict(
            self.testingSamples,
            batch_size=self.batchSize,
            verbose=0
        )
        
    def PlotPredictions(self, bins: int=200) -> None:
        """Plots the predictions on the training, validation, and testing data.
        
        Parameters:
            bins (int):
                The number of bins to use when plotting the predictions.
        """

        # Plots the histogram of the prediction values.
        predictionLabels = ("Training", "Validation", "Testing")
        predictions = (self.trainingPredictions, self.validationPredictions, self.testingPredictions)

        for i in range(len(predictions)):
            plt.hist(predictions[i], bins=bins)
            plt.title(f"Distribution of Predictions on {predictionLabels[i]} Data")
            plt.xlabel("Prediction (0=SM, 1=BSM)")
            plt.ylabel("Frequency")
            plt.savefig(f"{self.predictionsOutputFolder}/{predictionLabels[i]} Prediction Distribution.png")
            plt.clf()
        
    def PlotRocCurve(self) -> None:
        """Plots the ROC curve on the training, validation, and testing data."""

        # Stores the labels and predictions in iterable objects.
        labels = (self.trainingLabels, self.validationLabels, self.testingLabels)
        predictions = (self.trainingPredictions, self.validationPredictions, self.testingPredictions)
        names = ("Training", "Validation", "Testing")

        # Loops through the types of data.
        for i in range(len(predictions)):
            # Gets the false positive rate, true positive rate, and the corresponding thresholds.
            fpr, tpr, thresholds = roc_curve(labels[i], predictions[i])
        
            # Computes the area under the curve.
            roc_auc = auc(fpr, tpr)
        
            # Plots the ROC curve and saves it to file.
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Receiver Operating Characteristic ({names[i]})')
            plt.legend(loc="lower right")
            plt.savefig(f"{self.outputFolder}/{names[i]} ROC Curve.png")

    def LoadModel(self, modelPath: str) -> None:
        """Loads a model in from a file.

        Also sets every other variable which stores model data
        in the instance to None, since no information is determined
        from the loaded model.
        
        Parameters:
            modelPath (str):
                The file path that the model is located at.
        """

        self.model = load_model(modelPath)

        # Undefines everything about the class which contains model info,
        # since no information is known about this new model.
        # The output folders and other such things are kept the same.
        # NOTE: Need to figure out what information can be obtained from a loaded model.
        self.nodes = None
        self.epochs = None
        self.batchSize = None
        self.learningRate = None
        self.history = None
        self.validationSplit = None
        self.testingSplit = None
        self.inputDataNames = None
        self.trainingSamples = None
        self.trainingLabels = None
        self.validationSamples = None
        self.validationLabels = None
        self.testingSamples = None
        self.testingLabels = None
        self.trainingPredictions = None
        self.validationPredictions = None
        self.testingPredictions = None
        self.inputDataNames = None

    def TuneHyperParameters(self, epochs: int=20, maxTrials: int=25, executionsPerTrial: int=1, tuningSubfolder: str="Tuning Results"):
        """Tunes the hyperparameters to find the best options to maximise accuracy and/or minimise loss.
        
        Assumes that the input data has already been given to the instance.
        
        Parameters:
            epochs (int):
                The number of epochs to train each model for.
            maxTrials (int):
                The maxmimum number of trials to run.
            executionsPerTrial (int):
                The number of networks to run per trial. Larger numbers require more
                time but minimise random variance in the results.
            tuningSubFolder (str):
                The name (NOT the path) of the subfolder in which to store the
                tuning outputs.
        """

        # Creates and stores the tuning subdirectory.
        self.tuningOutputFolder = f"{self.outputFolder}/{tuningSubfolder}"
        os.mkdir(self.tuningOutputFolder)

        # Creates the tuner.
        tuner = RandomSearch(
            hypermodel=NeuralNetwork._BuildModelWithHyperparameters,
            objective="val_accuracy",
            max_trials=maxTrials,
            executions_per_trial=executionsPerTrial,
            overwrite=True,
            directory=self.outputFolder,
            project_name=tuningSubfolder
        )

        # Prints a summary of the search space.
        tuner.search_space_summary()

        # Searches the hyperparameter space using the training data.
        tuner.search(self.trainingSamples,
                     self.trainingLabels,
                     epochs=epochs,
                     validation_data=(self.validationSamples, self.validationLabels))
        
        # Prints a summary of the search results.
        tuner.results_summary()

        # Gets the three best performing models.
        bestModels = tuner.get_best_models(num_models=3)
        bestModels[0].summary()

    def _BuildModelWithHyperparameters(hp: HyperParameters) -> Sequential:
        """Creates a model with hyperparameters that can then be tuned with
        Keras Tuner.
        
        Parameters:
            hp: HyperParameters
                Contains the hyperparameters.
                
        Returns:
            Sequential:
                A sequential model with the desired hyperparameters.
        """

        # Creates the sequential model.
        model = Sequential()

        # Tunes the number of layers.       
        for layer in range(hp.Int("numLayers", 2, 5)):
            model.add(Dense(
                # Tunes the number of nodes in each layer.
                units=hp.Int(f"nodes_{layer}", min_value=24, max_value=192, step=24),
                activation='relu'
            ))

        # Adds the output layer.
        model.add(Dense(1, activation='sigmoid'))
        
        # Compiles the model with a tuned learning rate.
        model.compile(
            optimizer=Adam(learning_rate=hp.Float("learningRate", min_value=0.0001, max_value=0.01, sampling='log')),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def PlotHistoryFromFile(fileName: str, outputFolder: str="."):
        """Plots the accuracy."""
        history = np.load(fileName, allow_pickle=True).item()
 
        # Gets the accuracy and loss values over time during training.
        trainingAccuracy = history.history['accuracy']
        validationAccuracy = history.history['val_accuracy']
        trainingLoss = history.history['loss']
        validationLoss = history.history['val_loss']
        epochsRange = range(1, len(trainingAccuracy) + 1)

        fontSize = 12

        # Plots the accuracy over time.
        plt.plot(epochsRange, trainingAccuracy, label="Training Accuracy")
        plt.plot(epochsRange, validationAccuracy, label="Validation Accuracy")
        plt.title("Training and Validation Accuracy", fontsize=fontSize)
        plt.xlabel("Epochs", fontsize=fontSize)
        plt.ylabel("Accuracy (%)", fontsize=fontSize)
        plt.legend()
        plt.savefig(f"{outputFolder}/Accuracy over Time.png")
        plt.clf()

        # Plots the loss over time.
        plt.plot(epochsRange, trainingLoss, label="Training Loss")
        plt.plot(epochsRange, validationLoss, label="Validation Loss")
        plt.title("Training and Validation Loss", fontsize=fontSize)
        plt.xlabel("Epochs", fontsize=fontSize)
        plt.ylabel("Loss", fontsize=fontSize)
        plt.legend()
        plt.savefig(f"{outputFolder}/Loss over Time.png")
        plt.clf()