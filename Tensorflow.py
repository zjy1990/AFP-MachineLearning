

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import raw datam


#financial
# raw_data = pd.read_csv('data/fin_stock.csv',sep = ',')
# train_data = raw_data.iloc[0:1511,]
# test_data = raw_data.iloc[1481:1761,]

#Index
# raw_data = pd.read_csv('data/IDX_sp500_only1.csv',sep = ',')
# train_data = raw_data.iloc[0:5000,]
# test_data = raw_data.iloc[5200:5600,]
#tech firm
raw_data = pd.read_csv('data/GOOG.csv',sep = ',')
train_data = raw_data.iloc[0:35000,]
test_data = raw_data.iloc[35000:36645,]
#params
batch_size = 100
num_per_batch = train_data.shape[1] - 2
num_of_time_series = 100
num_class = 4
lstm_size = 64
num_iteration = 2000
#num_iteration = train_data.shape[0] - batch_size + 1
display_step = batch_size
#strategy params
target_buy = 0.00001
target_sell = -0.00001
trans_cost = 0.000
borrow_rate = 0.000
initial_capital = 100
ptf_value = []
ptf_value.append(initial_capital)
ptf_ret = []

#function to select batch data for random draw
def getTrainingBatch_random(batch_size, traindata):
    maxNumber = traindata.shape[0] - num_of_time_series - 1
    batchIndex = np.random.randint(0, maxNumber, batch_size)
    trainBatch = np.ndarray((batch_size, num_per_batch, num_of_time_series))
    trainLabel = pd.DataFrame(data=np.zeros((batch_size, num_class)))

    for i in range(len(batchIndex)):
        trainBatch[i,] = (traindata.iloc[batchIndex[i]:(batchIndex[i] + num_of_time_series), 2:traindata.shape[1]]).transpose()
        trainLabel.loc[i, 0] = np.int(traindata.iloc[(batchIndex[i] + num_of_time_series - 1), 1] >= target_buy)
        trainLabel.loc[i, 1] = np.int((traindata.iloc[(batchIndex[i] + num_of_time_series - 1), 1] < target_buy) & (traindata.iloc[(batchIndex[i] + num_of_time_series - 1), 1] >= 0))
        trainLabel.loc[i, 2] = np.int((traindata.iloc[(batchIndex[i] + num_of_time_series - 1), 1] < 0) & (traindata.iloc[(batchIndex[i] + num_of_time_series - 1), 1] >= target_sell))
        trainLabel.loc[i, 3] = np.int(traindata.iloc[(batchIndex[i] + num_of_time_series - 1), 1] < target_sell)

    trainBatch = trainBatch.tolist()
    return(trainBatch,trainLabel)


def getTestingBatch_timeseries(batch_size, testdata):
    real_return = testdata.iloc[-1, 1]
    testLabel = pd.DataFrame(data=np.zeros((batch_size, num_class)))

    testBatch = np.ndarray((batch_size, num_per_batch, num_of_time_series))

    for i in range(batch_size):
        testBatch[i,] = (testdata.iloc[0:num_of_time_series, 2:traindata.shape[1]]).transpose()
        testLabel.loc[i, 0] = np.int(real_return >= target_buy)
        testLabel.loc[i, 1] = np.int((real_return < target_buy) & (real_return >= 0))
        testLabel.loc[i, 2] = np.int((real_return < 0) & (real_return >= target_sell))
        testLabel.loc[i, 3] = np.int(real_return < target_sell)

    testBatch = testBatch.tolist()

    return(testBatch,testLabel,real_return)

def getReturn(net_position, action, actual_return):

    if action == "Buy":
        if net_position == 0:
            Tcost = trans_cost
        elif net_position == 1:
            Tcost = 0
        else:
            Tcost = trans_cost*2
        net_position = 1
        adj_ret = 1 + actual_return - Tcost

    elif action == "Sell":
        if net_position == 0:
            Tcost = trans_cost + borrow_rate
        elif net_position == 1:
            Tcost = 2*trans_cost + borrow_rate
        else:
            Tcost = borrow_rate
        net_position = -1
        adj_ret = 1 - actual_return - Tcost
    elif action == "Hold+":
        if net_position == -1:
            Tcost = trans_cost
            adj_ret = 1 - Tcost
            net_position = 0
        elif net_position == 1:
            Tcost = 0
            adj_ret = 1 + actual_return
            net_position = 1
        else:
            adj_ret = 1
            net_position = 0
    else:
        if net_position == -1:
            Tcost = borrow_rate
            adj_ret = 1 - actual_return - Tcost
            net_position = -1
        elif net_position == 1:
            Tcost = trans_cost
            adj_ret = 1 - Tcost
            net_position = 0
        else:
            adj_ret = 1
            net_position = 0

    return(adj_ret,net_position)

#define weight and bias
weight = tf.Variable(tf.truncated_normal([lstm_size,num_class]))
bias = tf.Variable(tf.constant(0.1,shape=[num_class]))
#define labels and input data format
labels = tf.placeholder(tf.float32,[batch_size,num_class])
input_data = tf.placeholder(tf.float32, [batch_size, num_per_batch, num_of_time_series])

#LSTM cell construction
def LSTM(input_data,weight,bias):
    input_data = tf.unstack(input_data, axis = 1)
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
        #traindata = train_data.iloc[step:step+batch_size,:]
        traindata =train_data
        nextTrainBatch,nextTrainBatchLabels = getTrainingBatch_random(batch_size,traindata)
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
    # #testing data
    for step in range(test_data.shape[0] - num_of_time_series - 1):

        nextTestBatch,nextTestBatchLabels,realize_return = getTestingBatch_timeseries(batch_size, test_data.iloc[step : step + num_of_time_series , :])
        #nextBatch = tf.unstack(nextBatch)
        sess.run(optimizer, feed_dict={input_data: nextTestBatch, labels: nextTestBatchLabels})

        pred_result = sess.run(prediction_results, feed_dict={input_data: nextTestBatch, labels:nextTestBatchLabels})

        if pred_result == 0:
            action = "Buy"
        elif pred_result == 1:
            action = "Hold+"
        elif pred_result == 2:
            action = "Hold-"
        else:
            action = "Sell"

        date = test_data.iloc[step + num_of_time_series+1,0]
        adj_ret,net_position = getReturn(net_position,action,realize_return)
        ptf_value.append(adj_ret*ptf_value[step])
        ptf_ret.append(adj_ret-1)
        print(str(date) +" " + action +" : Cumulative portfolio value = " + str(ptf_value[step+1]) + " " + str(realize_return))

            # Calculate batch accuracy & loss
        # acc, loss = sess.run([accuracy, cost], feed_dict={input_data: nextTestBatch, labels: nextTestBatchLabels})
        # print("Minibatch Loss= " + \
        #           "{:.6f}".format(loss) + ", Training Accuracy= " + \
        #           "{:.5f}".format(acc))

    print("Testing Finished!")



benchmark = (initial_capital * np.cumprod(1 + test_data.iloc[num_of_time_series-1:test_data.shape[0], 1])).tolist()
SR_ptf = np.average(ptf_ret)/np.std(ptf_ret)*np.sqrt(252)
SR_mkt = np.average(test_data.iloc[num_of_time_series:test_data.shape[0], 1]) / np.std(test_data.iloc[num_of_time_series:test_data.shape[0], 1]) * np.sqrt(252)
print("Portfolio sharpe ratio = "+ str(SR_ptf))
print("Market sharpe ratio = "+ str(SR_mkt))
plt.plot(ptf_value,'-b',label = 'Portfolio')
plt.plot(benchmark,'-r',label = 'Benchmark')
plt.axis([0, test_data.shape[0],np.min(ptf_value)*0.9,np.max(ptf_value)*1.1])
plt.ylabel('Cumulative portfolio value')
plt.xlabel('Time')
plt.legend()
plt.show()