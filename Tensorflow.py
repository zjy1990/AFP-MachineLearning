

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import raw data
#financial
#train_data = pd.read_csv('data/fin_stock.csv',sep = ',')
#test_data = pd.read_csv('data/test_fin.csv',sep = ',')
#Index
raw_data = pd.read_csv('data/index_data.csv',sep = ',')
train_data = raw_data.iloc[0:3164,]
test_data = raw_data.iloc[3165:3382,]
#tech firm
# train_data = pd.read_csv('data/tech_stock.csv',sep = ',')
# test_data = pd.read_csv('data/test_tech.csv',sep = ',')

#params
batch_size = 100
num_per_batch = train_data.shape[1] - 2
num_class = 3
lstm_size = 64
#num_iteration = 5000
num_iteration = train_data.shape[0] - batch_size + 1
display_step = batch_size
#strategy params
target_buy = 0.003
target_sell = -0.004
trans_cost = 0.001
borrow_rate = 0.0002
initial_capital = 100
ptf_value = []
ptf_value.append(initial_capital)


#function to select batch data for random draw
def getTrainingBatch_random(batch_size, traindata):
    maxNumber = traindata.shape[0]
    batchIndex = np.random.randint(0, maxNumber, batch_size)
    trainBatch = traindata.iloc[batchIndex, :]
    trainLabel = pd.DataFrame(data=np.zeros((batch_size, num_class)))
    trainLabel.loc[:, 0] = np.int_(trainBatch.iloc[:, 1] >= target_buy)
    trainLabel.loc[:, 1] = np.int_((trainBatch.iloc[:, 1] < target_buy) & (trainBatch.iloc[:, 1] >= target_sell))
    trainLabel.loc[:, 2] = np.int_(trainBatch.iloc[:, 1] < target_sell)
    trainBatch = trainBatch.iloc[:, 2:trainBatch.shape[1]]
    trainBatch = np.array(trainBatch)
    trainBatch = np.reshape(trainBatch,(batch_size,num_per_batch,1)).tolist()
    return(trainBatch,trainLabel)


#function to select batch data for random draw
def getTrainingBatch_timeseries(batch_size, traindata):
    trainBatch = traindata
    trainLabel = pd.DataFrame(data=np.zeros((batch_size, num_class)))
    trainLabel.loc[:, 0] = np.int_(trainBatch.iloc[:, 1] >= target_buy)
    trainLabel.loc[:, 1] = np.int_((trainBatch.iloc[:, 1] < target_buy) & (trainBatch.iloc[:, 1] >= target_sell))
    trainLabel.loc[:, 2] = np.int_(trainBatch.iloc[:, 1] < target_sell)
    trainBatch = trainBatch.iloc[:, 2:trainBatch.shape[1]]
    trainBatch = np.array(trainBatch)
    trainBatch = np.reshape(trainBatch,(batch_size,num_per_batch,1)).tolist()
    return(trainBatch,trainLabel)

def getTestingBatch_timeseries(batch_size, testdata):
    testdata = testdata.values.reshape((1,test_data.shape[1]))
    testLabel = pd.DataFrame(data=np.zeros((batch_size, num_class)))
    real_return = testdata[0, 1]
    testLabel.loc[:, 0] = np.int_(real_return >= target_buy)
    testLabel.loc[:, 1] = np.int_((real_return < target_buy) & (real_return >= target_sell))
    testLabel.loc[:, 2] = np.int_(real_return < target_sell)

    testBatch = pd.DataFrame(data = np.zeros((batch_size,num_per_batch)))
    testBatch = np.repeat(testdata[:, 2:testdata.shape[1]],batch_size,axis=0)
    testBatch = np.reshape(testBatch,(batch_size,num_per_batch,1)).tolist()

    return(testBatch,testLabel,real_return)

def getReturn(net_position,action,realize_return):

    if action == "Buy":
        if net_position == 0:
            Tcost = trans_cost
        elif net_position == 1:
            Tcost = 0
        else:
            Tcost = trans_cost*2
        net_position = 1
        adj_ret = 1 + realize_return - Tcost
    elif action == "Sell":
        if net_position == 0:
            Tcost = trans_cost + borrow_rate
        elif net_position == 1:
            Tcost = 2*trans_cost + borrow_rate
        else:
            Tcost = 0
        net_position = -1
        adj_ret = 1 - realize_return - Tcost
    else:
        if net_position == -1:
            Tcost = borrow_rate
            adj_ret = 1 - realize_return - Tcost
        else:
            Tcost = 0
            adj_ret = 1 + realize_return - Tcost

    return(adj_ret,net_position)

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
prediction_results = tf.argmax(prediction,1)[0]
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)


#run model random time series
    print("Optimization Starts!")
    for step in range(num_iteration):
        traindata = train_data.iloc[step:step+batch_size,:]
        nextTrainBatch,nextTrainBatchLabels = getTrainingBatch_timeseries(batch_size,traindata)
        #nextBatch = tf.unstack(nextBatch)
        sess.run(optimizer,feed_dict= {input_data: nextTrainBatch,labels: nextTrainBatchLabels})
        if step % display_step == 0:#report summary
            # Calculate batch accuracy & loss
            acc, loss = sess.run([accuracy, cost], feed_dict={input_data: nextTrainBatch, labels: nextTrainBatchLabels})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))

    print("Optimization Finished!")

    net_position = 0 #initialize action
    #testing data
    for step in range(test_data.shape[0]):

        nextTestBatch,nextTestBatchLabels,realize_return = getTestingBatch_timeseries(batch_size, test_data.iloc[step,:])
        #nextBatch = tf.unstack(nextBatch)
        sess.run(optimizer, feed_dict={input_data: nextTestBatch, labels: nextTestBatchLabels})

        pred_result = sess.run(prediction_results, feed_dict={input_data: nextTestBatch, labels:nextTestBatchLabels})
        if pred_result == 0:
            action = "Buy"
        elif pred_result == 1:
            action = "Hold"
        else:
            action = "Sell"
        date = test_data.iloc[step,0]
        adj_ret,net_position = getReturn(net_position,action,realize_return)
        ptf_value.append(adj_ret*ptf_value[step])
        print(str(date) +" " + action +" : Cumulative portfolio value = " + str(ptf_value[step+1]))

    print("Testing Finished!")

benchmark = initial_capital*np.cumprod(1+test_data.iloc[:, 1])
plt.plot(ptf_value)
plt.plot(benchmark)
plt.ylabel('Cumulative portfolio value $')
plt.xlabel('Time')
plt.show()