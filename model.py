from argparse import ArgumentParser
from pathlib import Path
from pickle import dump, load
from typing import NoReturn

import matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import datapreprocessing


class KerasTrainer:
    """
    This class process the data, trains an Deep Learning Model.
    """

    def __init__(self, args: ArgumentParser) -> NoReturn:
        # Create the args object
        self._args = args
        """Init method.
        Args:
            args (ArgumentParser): The arguments of the training and testing session.
        Returns:
            NoReturn: This method does not return anything.
        """
        # Process data
        self.xs, self.ys = datapreprocessing.DataPipeline(args=args)
        self.path = f"./saved_model/Keras/model.pkl"
        self.saved_model = self._load_model()

    def _train(self) -> NoReturn:
        """This function trains a model.
        Returns:
            NoReturn: This function does not return anything.
        """
        # Declare a saving path
        self.model_path = f"./saved_model/Keras/"
        # Create the saving path if it does not exist
        Path(self.model_path).mkdir(parents=True, exist_ok=True)

        # Create and train model
        self.model = Sequential()
        self.model.add(Embedding(26717, 64, input_length=222))  # 61983
        self.model.add(Bidirectional(LSTM(20)))
        self.model.add(Dense(26717, activation="softmax"))
        self.model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        self.history = self.model.fit(self.xs, self.ys, epochs=100, verbose=1)

        if self._args.visualize_results:
            # Plot Results
            Path(f"./visualization_results/").mkdir(parents=True, exist_ok=True)
            plt.plot(self.history["acc"])
            plt.xlabel("Epochs")
            plt.ylabel("acc")
            plt.savefig(
                f"./visualization_results/results.png",
                bbox_inches="tight",
            )
        # Save the model
        dump(obj=self.model, file=open(f"{self.model_path}/model.pkl", "wb"))

    def test(self, token_list):
        self.preds = self.saved_model.predict_classes(token_list)
        return self.preds

    def _load_model(self):
        """This method loads an XGBoost Regressor model from the given path.
        Returns:
            XGBRegressor: The loaded version of XGBoost Regressor.
        """
        # Load the model, then return
        return load(open(f"{self.path}", "rb"))
