import tensorflow as tf
import numpy as np
import glob #this will be useful when reading reviews from file
import os
import tarfile
import string
import re
from _collections import defaultdict


batch_size = 50
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def load_data(glove_dict):
    """
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form"""
    if os.path.exists(os.path.join(os.path.dirname(__file__), "data.npy")):
        print("loading saved parsed data, to reparse, delete 'data.npy'")
        data = np.load("data.npy")
        return data
    else:
        filename = check_file('reviews.tar.gz',14839260)
        extract_data(filename)
        dir= os.path.dirname(__file__)

        files= glob.glob(os.path.join(dir,
                                        'data2/pos/*'))
        files.extend(glob.glob(os.path.join(dir,
                                        'data2/neg/*')))

        data = []

        for f in files:
            with open(f, 'r') as openf:
                s = openf.read()
                clean_line(s)
                words = s.split(" ")
                for word in words:
                    if(word in string.punctuation or any(char.isdigit() for char in word)):
                        continue
                    data.append(word)
          
        print(data[:5])
        # np.save("data", data)
        return data


def load_glove_embeddings():
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.50d.txt"
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119"}
    """
    data = open("glove.6B.50d.txt",'r',encoding="utf-8")
    embeddings = []
    word_index_dict = defaultdict()
    index = 0
    for lines in data:
        wordVector = lines.split(" ")
        if(wordVector[0] in string.punctuation or any(char.isdigit() for char in wordVector[0])):
            continue
        embeddings.append(wordVector[1:-1])
        word_index_dict[wordVector[0]] = index
        index+=1
    return embeddings, word_index_dict


def define_graph(glove_embeddings_arr):
    """
    Define the tensorflow graph that forms your model. You must use at least
    one recurrent unit. The input placeholder should be of size [batch_size,
    40] as we are restricting each review to it's first 40 words. The
    following naming convention must be used:
        Input placeholder: name="input_data"
        labels placeholder: name="labels"
        accuracy tensor: name="accuracy"
        loss tensor: name="loss"

    RETURN: input placeholder, labels placeholder, dropout_keep_prob, optimizer, accuracy and loss
    tensors"""
    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss


def check_file(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        print("please make sure {0} exists in the current directory".format(filename))
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            "File {0} didn't have the expected size. Please ensure you have downloaded the assignment files correctly".format(filename))
    return filename

def extract_data(filename):
    """Extract data from tarball and store as list of strings"""
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'data2/')):
        with tarfile.open(filename, "r") as tarball:
            dir = os.path.dirname(__file__)
            tarball.extractall(os.path.join(dir, 'data2/'))
    return

def clean_line(string):

    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())