
import tensorflow as tf
import pandas as pd
import numpy as np

#params
batch_size = 30
num_per_batch = 10
num_class = 2
lstm_size = 128
num_iteration = 100

#fake data
stock_data = pd.DataFrame(data=np.random.randn(batch_size*num_iteration, num_per_batch)*0.1)
labels_data = pd.DataFrame(data=np.zeros((batch_size*num_iteration,num_class)))
labels_data.loc[:,0] = np.int_(stock_data.loc[:,0]>=0)
labels_data.loc[:,1] = np.int_(stock_data.loc[:,0]<0)
stock_data = stock_data.values.tolist()

weight = tf.Variable(tf.truncated_normal([lstm_size,num_class]))
bias = tf.Variable(tf.constant(0.1,shape=[num_class]))
labels = tf.placeholder(tf.float32,[batch_size,num_class])
input_data = tf.placeholder(tf.float32,[batch_size,num_per_batch,1])


def LSTM(input_data,weight,bias):
    input_data = tf.unstack(input_data, num_per_batch, 1)
    lstmCell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    lstmCell = tf.nn.rnn_cell.DropoutWrapper(cell=lstmCell, output_keep_prob=0.85)
    value, _ = tf.nn.static_rnn(lstmCell, input_data, dtype=tf.float32)
    value = tf.stack(value)
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)
    return prediction

prediction = LSTM(input_data,weight,bias)
#LSTM tensorflow models

#input_data = tf.unstack(input_data,num_per_batch,1)







#value = tf.transpose(value,[1,0,2])



loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels = labels))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init)
    #save summary
    # tf.summary.scalar('Loss',loss)
    # tf.summary.scalar('Accuracy',accuracy)
    # tf.summary.histogram('Weight',weight)
    # tf.summary.histogram('Bias',bias)
    # merged_summary = tf.summary.merge_all()
    # logdir = "~/Desktop"
    # writer = tf.summary.FileWriter(logdir)
    # writer.add_graph(sess.graph)

#run model
    for i in range(num_iteration):
        nextBatch = stock_data[i*batch_size:(i+1)*batch_size-1]
        nextBatchLabels = labels_data.loc[i*batch_size:(i+1)*batch_size-1,]
        nextBatch = tf.unstack(nextBatch)
        sess.run(optimizer,feed_dict= {input_data: nextBatch,labels: nextBatchLabels})
