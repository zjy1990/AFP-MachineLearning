import tensorflow as tf
import pandas as pd
import numpy as np

#params
batchSize = 30
iterations = 100
numClass = 3
lstm_size = ?

def getTrainBatch():
    batch = data[]

def getTestBatch():

labels = tf.placeholder(tf.float32,[batchSize,numClass])
input_data = tf.placeholder(tf.float32,[batchSize,])

lstmCell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
weight = tf.Variable(tf.truncated_normal([lstm_size,numClass]))

