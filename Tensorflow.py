
import tensorflow as tf
import pandas as pd
import numpy as np


stock_data = pd.DataFrame(data=np.random.randn(1000)*0.1)
stock_data['labels'] = 0
stock_data['labels'] = np.int_(stock_data.loc[:,0]>=0 )
#stock_data['labels'] = -np.int_(stock_data.loc[:,0]<=-0.05)



#params
batch_size = 10
num_class = 1
lstm_size = 64
num_interation = 100

#LSTM tensorflow models
labels = tf.placeholder(tf.float32,[batch_size,num_class])
input_data = tf.placeholder(tf.float32,[batch_size,1])

lstmCell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
lstmCell = tf.nn.rnn_cell.DropoutWrapper(cell=lstmCell,output_keep_prob=0.85)

value,_ = tf.nn.dynamic_rnn(lstmCell,input_data,dtype=tf.float32)
weight = tf.Variable(tf.truncated_normal([lstm_size,num_class]))
bias = tf.Variable(tf.constant(0.1,shape=[lstm_size,num_class]))

last = tf.gather(value,int(value.get_shape()[0])-1)
prediction = (tf.matmul(last,weight)+bias)


correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels = labels))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)


sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
#save summary
tf.summary.scalar('Loss',loss)
tf.summary.scalar('Accuracy',accuracy)
tf.summary.histogram('Weight',weight)
tf.summary.histogram('Bias',bias)
merged_summary = tf.summary.merge_all()
logdir = "/temp/lstm_8"
writer = tf.summary.FileWriter(logdir)
writer.add_graph(sess.graph)


#run model
for i in range(num_interation):
    nextBatch = stock_data[i*batch_size:(i+1)*batch_size,1]
    nextBatchLabels = stock_data[i*batch_size:(i+1)*batch_size,2]
    sess.run(optimizer,{input_data:nextBatch,labels: nextBatchLabels})
    if(i % 5 == 0):
        summary = sess.run(merged_summary,{input_data : nextBatch, labels : nextBatchLabels})
        writer.add_summary(summary,i)