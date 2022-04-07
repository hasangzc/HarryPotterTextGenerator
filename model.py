import tensorflow as tf
from argparse import ArgumentParser

from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from datapreprocessing import DataPipeline


xs, ys = DataPipeline(args=ArgumentParser)

model = Sequential()
model.add(Embedding(26717, 64, input_length=max_sequence_len - 1))
model.add(Bidirectional(LSTM(20)))
model.add(Dense(26717, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(xs, ys, epochs=500, verbose=1)
