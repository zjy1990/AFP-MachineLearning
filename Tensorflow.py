import tensorflow as tf
import pandas as pd
import numpy as np

#params
num_batch = 4
batch_size = 10
time_step = 5
lstm_size = 64
dataset = tf.placeholder(tf.float32,[num_batch,batch_size,num_feature])
lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)

hidden_state = tf.zeros([batch_size,lstm.state_size])
current_state = tf.zeros([batch_size,lstm.state_size])
state = hidden_state,current_state
probabilities = []

loss = 0.0
for current_batch in dataset