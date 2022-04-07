from argparse import ArgumentParser
from os import walk
from train import declareParserArguments

import numpy as np
import nltk
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def DataPipeline(args: ArgumentParser):
    # If not exist merge all books into the one text file named "allbooks"
    if "allbooks.txt" in file_list("./Data/allbooks.txt"):
        pass
    else:
        combine_books()

    if args.data_informations:
        information_about_books(args=args)
        print("Data informations ops finished!")

    # Seperate sentences and add a list
    sentences = separete_books_sentences("./Data/allbooks.txt")
    print("Sentences Ready!")

    #  Tokenize data
    tokenizer = Tokenizer(num_words=60000)
    tokenizer.fit_on_texts(sentences)
    world_index = tokenizer.word_index
    print("Tokenizer ops finished!")

    # Text Sequence
    sequences = tokenizer.texts_to_sequences(sentences)

    # Pad ops: same input length for  model
    padded = pad_sequences(sequences)
    input_seq = np.array(padded)

    # Let's label the last character
    inp = input_seq[:, :-1]
    labels = input_seq[:, -1]

    # One-hot-encode ops for label classification
    target = tf.keras.utils.to_categorical(labels, num_classes=(len(world_index) + 1))

    # Return inp and target
    return inp, target


def file_list(path):
    """Wİth this function File names in the folder can be listed
     Args:
    path: The path of the folder in which you will look at the file names
    Returns:
    List of filenames in Path
    """
    file_names = list()
    for root, dirc, files in walk(path):
        for FileName in files:
            file_names.append(FileName)
    return file_names


def combine_books():
    """With this function combines all books and creates a single txt file
    Returns:
    NoReturn
    """
    filenames = file_list("./Data")

    with open("./Data/allbooks.txt", "w") as outfile:
        for fname in filenames:
            # print(fname)
            with open(f"./Data/{fname}") as infile:
                for line in infile:
                    outfile.write(line)


def information_about_books(args: ArgumentParser):
    # read allboks.txt
    all_books = open("./Data/allbooks.txt", "r", encoding="utf-8")
    all_books = all_books.read()

    #  How many words are in txt? It doesn't just bring words. It will also count special characters.
    len_total_words = len(all_books.split())  # 1168853
    # How many different words are in txt?
    len_unique_words = len(set(all_books.split()))  # 60966
    # Check Unique characters in txt
    unique_characters = set(all_books)
    # How many unique charachter in txt
    len_unique_characters = len(set(all_books))

    # print informations:
    print(f"Number of words in books: {len_total_words}")
    print(f"Number of unique words in books: {len_unique_words}")
    print(f"Number of characters in books:{len_unique_characters}")
    print(f"Unique characters in books: {unique_characters}")


def separete_books_sentences(path):
    """With this function Separates sentences from all books and adds them to a list
    Args:
    path: The path of your text file
    Returns:
    sents: Sentences list
    """
    with open(path, "r") as in_file:
        text = in_file.read()
        sents = nltk.sent_tokenize(text)
        # fix some problems for data
        sents = [x.replace("\n", "") for x in sents]
        sents = [x.replace("Page", "") for x in sents]
        sents = [x.replace(" | ", "") for x in sents]
    return sents


parser = ArgumentParser(description="Harry Potter Text Generator")
# Add the arguments
args = declareParserArguments(parser=parser)
# DataPipeline(args=args)
