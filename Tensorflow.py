
import tensorflow as tf
import pandas as pd
import numpy as np

#params
batch_size = 100
num_per_batch = 20
num_class = 2
lstm_size = 64
num_iteration = 1000
display_step = 100

#fake data
stock_data = (np.random.randn(batch_size*num_iteration, num_per_batch,1)*0.1).tolist()
labels_data = pd.DataFrame(data=np.zeros((batch_size*num_iteration,num_class)))
predict_stock_data = pd.DataFrame(np.random.randn(batch_size*num_iteration,1))
labels_data.loc[:,0] = np.int_(predict_stock_data.loc[:,0]>=0)
labels_data.loc[:,1] = np.int_(predict_stock_data.loc[:,0]<0)

#define weight and bias
weight = tf.Variable(tf.truncated_normal([lstm_size,num_class]))
bias = tf.Variable(tf.constant(0.1,shape=[num_class]))
#define labels and input data format
labels = tf.placeholder(tf.float32,[batch_size,num_class])
input_data = tf.placeholder(tf.float32,[batch_size,num_per_batch,1])

#LSTM cell construction
def LSTM(input_data,weight,bias):
    input_data = tf.unstack(input_data, num_per_batch, 1)
    lstmCell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    lstmCell = tf.nn.rnn_cell.DropoutWrapper(cell=lstmCell, output_keep_prob=0.85)
    value, _ = tf.nn.static_rnn(lstmCell, input_data, dtype=tf.float32)
    value = tf.stack(value)
    last = tf.gather(value, int(value.get_shape()[0]) - 1) #take the last one
    prediction = (tf.matmul(last, weight) + bias)
    return prediction

prediction = LSTM(input_data,weight,bias)
#define cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels = labels))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)
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
    for step in range(num_iteration):
        nextBatch = stock_data[step*batch_size:(step+1)*batch_size]
        nextBatchLabels = labels_data.loc[step*batch_size:(step+1)*batch_size-1,]
        #nextBatch = tf.unstack(nextBatch)
        sess.run(optimizer,feed_dict= {input_data: nextBatch,labels: nextBatchLabels})
        if step % display_step == 0 or step == 1:#report summary
            # Calculate batch accuracy & loss
            acc, loss = sess.run([accuracy, cost], feed_dict={input_data: nextBatch, labels: nextBatchLabels})
            print("Step " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))

    print("Optimization Finished!")