import os
import string
import tempfile
import numpy as np 
import tensorflow as tf 

from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.preprocessing import sequence
from tensorboard import summary as summary_lib

#imdb = keras.datasets.imdb
(train_x, train_y), (test_x, test_y) = keras.datasets.imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}".fomat(len(train_x), len(train_y)))

print(train_x[0])

print('len: ', len(train_x[0], len(train_x[1])))

