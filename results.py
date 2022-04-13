import matplotlib.pyplot as plt
from argparse import ArgumentParser
from datapreprocessing import DataPipeline
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import model
import train


if __name__ == "__main__":
    # Declare an ArgumentParser object
    parser = ArgumentParser(description="TextGenerator")
    # Add the arguments
    args = train.declareParserArguments(parser=parser)

    # Create a KerasTrainer object
    trainer = model.KerasTrainer(args=args)

    seed_text = "Laurence went to dublin"
    next_words = 100

    for _ in range(next_words):
        # token_list = DataPipeline.tokenizer.texts_to_sequences([seed_text])[0]
        #  Tokenize data
        tokenizer = Tokenizer(num_words=60000)
        tokenizer.fit_on_texts(seed_text)
        world_index = tokenizer.word_index
        print("Tokenizer ops finished!")
        # Text Sequence
        token_list = tokenizer.texts_to_sequences(seed_text)
        token_list = pad_sequences(token_list, maxlen=222, padding="pre")
        predicted = trainer.test(token_list=token_list)
        output_word = ""
        for word, index in DataPipeline.tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    print(seed_text)
