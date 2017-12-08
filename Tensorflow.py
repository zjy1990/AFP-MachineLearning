import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import raw datam

#set training and testing data period format = 'yyyy-mm-dd'
train_date_end = '2007-12-31'
test_data_start = '2008-01-01'
test_data_end = '2008-12-31'


<<<<<<< HEAD
#set training and testing data period format = 'yyyy-mm-dd'
train_date_end = '1994-12-31'
test_data_start = '1995-01-01' # need to be date whereby the trading start date - batch_size
test_data_end = '1995-12-31'


batch_size = 240

#Index
raw_data = pd.read_csv('data/Index_data_stdized.csv',sep = ',')
train_data = raw_data[raw_data.Date <= train_date_end]
test_data = raw_data.iloc[(raw_data.index[raw_data['Date'] >= test_data_start])[0] - (batch_size - 1) : raw_data.index[raw_data['Date'] <= test_data_end][-1],:]
#tech firm
# raw_data = pd.read_csv('data/tech_stock.csv',sep = ',')
# train_data = raw_data[raw_data.Date <= train_date_end]
# test_data = raw_data[(raw_data.Date >= test_data_start)&(raw_data.Date <= test_data_end)]


#params
=======

#Index
raw_data = pd.read_csv('data/IDX_non_lag_asian_&_stock.csv',sep = ',')
train_data = raw_data[raw_data.Date <= train_date_end]
test_data = raw_data[(raw_data.Date >= test_data_start)&(raw_data.Date <= test_data_end)]


#params
batch_size = 240
>>>>>>> d306bd69d087d920577a4daa379b0f0a81edca5d
num_per_batch = train_data.shape[1] - 2
num_of_time_series = 1
num_class = 4
lstm_size = 100
<<<<<<< HEAD
=======
#num_iteration = 2500
>>>>>>> d306bd69d087d920577a4daa379b0f0a81edca5d
num_iteration = train_data.shape[0] - batch_size + 1
display_step = batch_size
#strategy params
target_buy = 0.003
target_sell = -0.003
<<<<<<< HEAD
trans_cost = 0.0010
=======
trans_cost = 0.001
>>>>>>> d306bd69d087d920577a4daa379b0f0a81edca5d
borrow_rate = 0.0002
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

def getTrainingBatch_timeseries(batch_size, traindata):
    trainBatch = np.ndarray((batch_size, num_per_batch, num_of_time_series))
    trainLabel = pd.DataFrame(data=np.zeros((batch_size, num_class)))

    for i in range(batch_size):
        trainBatch[i,] = (traindata.iloc[i:(i + num_of_time_series), 2:traindata.shape[1]]).transpose()
        trainLabel.loc[i, 0] = np.int(traindata.iloc[(i + num_of_time_series - 1), 1] >= target_buy)
        trainLabel.loc[i, 1] = np.int((traindata.iloc[(i + num_of_time_series - 1), 1] < target_buy) & (traindata.iloc[(i + num_of_time_series - 1), 1] >= 0))
<<<<<<< HEAD
        trainLabel.loc[i, 2] = np.int((traindata.iloc[(i + num_of_time_series - 1), 1] < 0) & (traindata.iloc[(i + num_of_time_series - 1), 1] >=target_sell))
        trainLabel.loc[i, 3] = np.int(traindata.iloc[(i + num_of_time_series - 1), 1] < target_sell)

    trainBatch = trainBatch.tolist()

    return(trainBatch,trainLabel)
=======
        trainLabel.loc[i, 2] = np.int((traindata.iloc[(i + num_of_time_series - 1), 1] < 0) & (traindata.iloc[(i + num_of_time_series - 1), 1] >= target_sell))
        trainLabel.loc[i, 3] = np.int(traindata.iloc[(i + num_of_time_series - 1), 1] < target_sell)

    trainBatch = trainBatch.tolist()
    return(trainBatch,trainLabel)

>>>>>>> d306bd69d087d920577a4daa379b0f0a81edca5d

def getTestingBatch_timeseries(batch_size, testdata):
    real_return = testdata.iloc[-1, 1]
    testLabel = pd.DataFrame(data=np.zeros((batch_size, num_class)))
    testBatch = np.ndarray((batch_size, num_per_batch, num_of_time_series))

    for i in range(batch_size):
        testBatch[i,] = (testdata.iloc[i:(i + num_of_time_series), 2:testdata.shape[1]]).transpose()
        testLabel.loc[i, 0] = np.int(testLabel.iloc[(i + num_of_time_series - 1), 1] >= target_buy)
        testLabel.loc[i, 1] = np.int((testLabel.iloc[(i + num_of_time_series - 1), 1] < target_buy) & (testLabel.iloc[(i + num_of_time_series - 1), 1] >= 0))
        testLabel.loc[i, 2] = np.int((testLabel.iloc[(i + num_of_time_series - 1), 1] < 0) & (testLabel.iloc[(i + num_of_time_series - 1), 1] >=target_sell))
        testLabel.loc[i, 3] = np.int(testLabel.iloc[(i + num_of_time_series - 1), 1] < target_sell)


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
    lstmCell = tf.nn.rnn_cell.DropoutWrapper(cell=lstmCell, output_keep_prob=0.80)
    value, _ = tf.nn.static_rnn(lstmCell, input_data, dtype=tf.float32)
    value = tf.stack(value)
    last = tf.gather(value, int(value.get_shape()[0]) - 1) #take the last one
    prediction = (tf.matmul(last, weight) + bias)
    return prediction

prediction = LSTM(input_data,weight,bias)
#define cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels = labels))
optimizer = tf.train.AdamOptimizer(1e-3).minimize(cost)
correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(labels,1))
prediction_results = tf.argmax(prediction,1)[0]
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)


#run model random time series
    print("Optimization Starts!")
    for step in range(num_iteration):
<<<<<<< HEAD
        traindata = train_data.iloc[step:step + batch_size, :]
=======
        #traindata = train_data.iloc[step:step+batch_size,:]
        traindata =train_data
>>>>>>> d306bd69d087d920577a4daa379b0f0a81edca5d
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
    accuracy_counter = 0  # initialize overall accuracy
    # #testing data
    for step in range(test_data.shape[0] - batch_size + 1):

        nextTestBatch, nextTestBatchLabels, realize_return = getTestingBatch_timeseries(batch_size, test_data.iloc[
                                                                                                    step: step + batch_size,
                                                                                                    :])
        #nextBatch = tf.unstack(nextBatch)
        pred_result = sess.run(prediction_results, feed_dict={input_data: nextTestBatch})
        sess.run(optimizer, feed_dict={input_data: nextTestBatch, labels: nextTestBatchLabels})

        if pred_result == 0:
            action = "Buy"
        elif pred_result == 1:
            action = "Hold+"
        elif pred_result == 2:
            action = "Hold-"
        else:
            action = "Sell"

<<<<<<< HEAD
        date = test_data.iloc[step + batch_size - 1, 0]
        adj_ret,net_position = getReturn(net_position,action,realize_return)
        ptf_value.append(adj_ret*ptf_value[step])
        ptf_ret.append(adj_ret-1)
=======
        date = test_data.iloc[step + num_of_time_series,0]
        adj_ret,net_position = getReturn(net_position,action,realize_return)
        ptf_value.append(adj_ret*ptf_value[step])
        ptf_ret.append(adj_ret-1)
        print(str(date) +" " + action +" : Cumulative portfolio value = " + str(ptf_value[step+1]))

>>>>>>> d306bd69d087d920577a4daa379b0f0a81edca5d
            # Calculate batch accuracy & loss
        # acc, loss = sess.run([accuracy, cost], feed_dict={input_data: nextTestBatch, labels: nextTestBatchLabels})
        # print("Minibatch Loss= " + \
        #           "{:.6f}".format(loss) + ", Training Accuracy= " + \
        #           "{:.5f}".format(acc))
<<<<<<< HEAD
        if (realize_return >= 0 and pred_result == 0) or (realize_return < 0 and pred_result == 1):
            accuracy_counter += 1
        print(str(date) + " " + action + " : Cumulative Strategy value = " + str(ptf_value[step + 1]))

    overall_accuracy = accuracy_counter / (test_data.shape[0] - batch_size + 1)
=======
        if (realize_return >= target_buy and pred_result == 0) or (realize_return < target_sell and pred_result == 3):
            accuracy_counter += 1
        print(str(date) + " " + action + " : Cumulative portfolio value = " + str(ptf_value[step + 1]))
        overall_accuracy = accuracy_counter / (test_data.shape[0] - num_of_time_series - 1)
>>>>>>> d306bd69d087d920577a4daa379b0f0a81edca5d
    print("Overall prediction accuracy: " + "{:.4f}".format(overall_accuracy))
    print("Testing Finished!")



benchmark = (initial_capital * np.cumprod(1 + test_data.iloc[batch_size-1:test_data.shape[0], 1])).tolist()
benchmark.insert(0,initial_capital)
ptf_volatility = np.std(ptf_ret)
SR_ptf = np.average(ptf_ret)/ptf_volatility*np.sqrt(252)
SR_mkt = np.average(test_data.iloc[batch_size -1 :test_data.shape[0], 1]) / np.std(test_data.iloc[batch_size -1 :test_data.shape[0], 1]) * np.sqrt(252)
# Maximum draw down
max_value = ptf_value[0]
min_value = ptf_value[0]
MDD = 0
for i in range(len(ptf_value) - 1):
    if(ptf_value[i+1] > max_value):
        max_value = ptf_value[i+1]
        min_value = ptf_value[i+1]
    elif(ptf_value[i+1] < min_value):
        min_value = ptf_value[i+1]
    MDD_temp = (min_value - max_value) / max_value
    if(MDD > MDD_temp):
        MDD = MDD_temp


print("Strategy sharpe ratio = "+ "{:.4f}".format(SR_ptf))
print("Strategy annualized volatility = "+ "{:.4f}".format(ptf_volatility*np.sqrt(252)))
print("Strategy Maximum Drawdown = "+ "{:.4f}".format(MDD))
print("Market sharpe ratio = "+ "{:.4f}".format(SR_mkt))



plt.plot(ptf_value,'-b',label = 'Strategy')
plt.plot(benchmark,'-r',label = 'Benchmark')
<<<<<<< HEAD
plt.axis([0, test_data.shape[0] - batch_size + 1,min(np.min(benchmark),np.min(ptf_value))*0.9,max(np.max(benchmark),np.max(ptf_value))*1.1])
plt.ylabel('Cumulative Strategy value')
=======
plt.axis([0, test_data.shape[0],min(np.min(benchmark),np.min(ptf_value))*0.9,max(np.max(benchmark),np.max(ptf_value))*1.1])
plt.ylabel('Cumulative portfolio value')
>>>>>>> d306bd69d087d920577a4daa379b0f0a81edca5d
plt.xlabel('Time')
plt.legend()
plt.show()