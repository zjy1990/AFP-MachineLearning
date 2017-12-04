
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import raw datam


#set training and testing data period format = 'yyyy-mm-dd'
train_date_end = '2007-12-31'
test_data_start = '2007-02-01' # need to be date whereby the trading start date - batch_size
test_data_end = '2008-12-31'



#Index
raw_data = pd.read_csv('data\Index_data_stdized.csv',sep = ',')
train_data = raw_data[raw_data.Date <= train_date_end]
test_data = raw_data[(raw_data.Date >= test_data_start)&(raw_data.Date <= test_data_end)]
#tech firm
# raw_data = pd.read_csv('data/tech_stock.csv',sep = ',')
# train_data = raw_data[raw_data.Date <= train_date_end]
# test_data = raw_data[(raw_data.Date >= test_data_start)&(raw_data.Date <= test_data_end)]

#params
batch_size = 240
num_per_batch = train_data.shape[1] - 2
num_of_time_series = 1
num_class = 2
lstm_size = 100
#num_iteration = 2000
num_iteration = train_data.shape[0] - batch_size + 1
display_step = batch_size
#strategy params
trans_cost = 0.0
borrow_rate = 0.0
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
        trainLabel.loc[i, 0] = np.int(traindata.iloc[(batchIndex[i] + num_of_time_series - 1), 1] >= 0)
        trainLabel.loc[i, 1] = np.int(traindata.iloc[(batchIndex[i] + num_of_time_series - 1), 1] < 0)

    trainBatch = trainBatch.tolist()
    return(trainBatch,trainLabel)

def getTrainingBatch_timeseries(batch_size, traindata):
    trainBatch = np.ndarray((batch_size, num_per_batch, num_of_time_series))
    trainLabel = pd.DataFrame(data=np.zeros((batch_size, num_class)))

    for i in range(batch_size):
        trainBatch[i,] = (traindata.iloc[i:(i + num_of_time_series), 2:traindata.shape[1]]).transpose()
        trainLabel.loc[i, 0] = np.int(traindata.iloc[(i + num_of_time_series - 1), 1] >= 0)
        trainLabel.loc[i, 1] = np.int(traindata.iloc[(i + num_of_time_series - 1), 1] < 0)

    trainBatch = trainBatch.tolist()
    return(trainBatch,trainLabel)

def getTestingBatch_timeseries(batch_size, testdata):
    real_return = testdata.iloc[-1, 1]
    testLabel = pd.DataFrame(data=np.zeros((batch_size, num_class)))
    testBatch = np.ndarray((batch_size, num_per_batch, num_of_time_series))

    for i in range(batch_size):
        testBatch[i,] = (testdata.iloc[i:(i + num_of_time_series), 2:testdata.shape[1]]).transpose()
        testLabel.loc[i, 0] = np.int(testdata.iloc[(i + num_of_time_series - 1), 1] >= 0)
        testLabel.loc[i, 1] = np.int(testdata.iloc[(i + num_of_time_series - 1), 1] < 0)

    testBatch = testBatch.tolist()

    return(testBatch,testLabel,real_return)

def getReturn(net_position, action, actual_return):

    if action == "Buy":
        if net_position == 1:
            Tcost = 0
        else:
            Tcost = trans_cost*2
        net_position = 1
        adj_ret = 1 + actual_return - Tcost

    elif action == "Sell":
        if net_position == 1:
            Tcost = 2*trans_cost + borrow_rate
        else:
            Tcost = borrow_rate
        net_position = -1
        adj_ret = 1 - actual_return - Tcost

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
optimizer = tf.train.AdamOptimizer(1e-3).minimize(cost)
correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(labels,1))
prediction_results = tf.argmax(prediction,1)[-1]
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


init = tf.global_variables_initializer()
Minibatch_loss = []
Minibatch_acc = []
with tf.Session() as sess:
    sess.run(init)


#run model random time series
    print("Optimization Starts!")
    for step in range(num_iteration):
        traindata = train_data.iloc[step:step+batch_size,:]
        nextTrainBatch,nextTrainBatchLabels = getTrainingBatch_timeseries(batch_size,traindata)
        # traindata =train_data
        # nextTrainBatch,nextTrainBatchLabels = getTrainingBatch_random(batch_size,traindata)
        sess.run(optimizer,feed_dict= {input_data: nextTrainBatch,labels: nextTrainBatchLabels})
        if step % display_step == 0:#report summary
            # Calculate batch accuracy & loss
            acc, loss = sess.run([accuracy, cost], feed_dict={input_data: nextTrainBatch, labels: nextTrainBatchLabels})
            Minibatch_loss.append(loss)
            Minibatch_acc.append(acc)
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))

    print("Optimization Finished!")

    net_position = 0 #initialize action
    accuracy_counter = 0 #initialize overall accuracy
    # #testing data
    print("Testing starts!")
    for step in range(test_data.shape[0] - batch_size + 1):

        nextTestBatch,nextTestBatchLabels,realize_return = getTestingBatch_timeseries(batch_size, test_data.iloc[step : step + batch_size , :])
        #nextBatch = tf.unstack(nextBatch)
        sess.run(optimizer, feed_dict={input_data: nextTestBatch, labels: nextTestBatchLabels})

        pred_result = sess.run(prediction_results, feed_dict={input_data: nextTestBatch, labels:nextTestBatchLabels})
        if pred_result == 0:
            action = "Buy"
        else:
            action = "Sell"

        date = test_data.iloc[step + batch_size - 1,0]
        adj_ret,net_position = getReturn(net_position,action,realize_return)
        ptf_value.append(adj_ret*ptf_value[step])
        ptf_ret.append(adj_ret-1)

            # Calculate batch accuracy & loss
        # acc, loss = sess.run([accuracy, cost], feed_dict={input_data: nextTestBatch, labels: nextTestBatchLabels})
        # print("Minibatch Loss= " + \
        #           "{:.6f}".format(loss) + ", Training Accuracy= " + \
        #           "{:.5f}".format(acc))
        if (realize_return >= 0 and pred_result == 0) or  (realize_return < 0 and pred_result == 1):
            accuracy_counter += 1
        print(str(date) +" " + action +" : Cumulative Strategy value = " + str(ptf_value[step+1]))
    overall_accuracy = accuracy_counter/(test_data.shape[0] - batch_size + 1)
    print("Overall prediction accuracy: " + "{:.4f}".format(overall_accuracy))
    print("Testing Finished!")



benchmark = (initial_capital * np.cumprod(1 + test_data.iloc[batch_size-1:test_data.shape[0], 1])).tolist()
benchmark.insert(0,initial_capital)
ptf_volatility = np.std(ptf_ret)
SR_ptf = np.average(ptf_ret)/ptf_volatility*np.sqrt(252)
SR_mkt = np.average(test_data.iloc[num_of_time_series-1:test_data.shape[0], 1]) / np.std(test_data.iloc[num_of_time_series-1 :test_data.shape[0], 1]) * np.sqrt(252)
# Maximum draw down
max_value = ptf_value[0]
min_value = ptf_value[0]
for i in range(len(ptf_value) - 1):
    if(ptf_value[i+1] > max_value):
        max_value = ptf_value[i+1]
        min_value = ptf_value[i+1]
    elif(ptf_value[i+1] < min_value):
        min_value = ptf_value[i+1]

MDD = (min_value - max_value)/max_value

print("Strategy sharpe ratio = "+ "{:.4f}".format(SR_ptf))
print("Strategy annualized volatility = "+ "{:.4f}".format(ptf_volatility*np.sqrt(252)))
print("Strategy Maximum Drawdown = "+ "{:.4f}".format(MDD))
print("Market sharpe ratio = "+ "{:.4f}".format(SR_mkt))



plt.plot(ptf_value,'-b',label = 'Strategy')
plt.plot(benchmark,'-r',label = 'Benchmark')
plt.axis([0, test_data.shape[0] - batch_size + 1,min(np.min(benchmark),np.min(ptf_value))*0.9,max(np.max(benchmark),np.max(ptf_value))*1.1])
plt.ylabel('Cumulative Strategy value')
plt.xlabel('Time')
plt.legend()
plt.show()

#plot loss and accuracy curve
# plt.plot(Minibatch_acc,label = 'Accuracy')
# plt.plot(Minibatch_loss,label = 'Loss')
# plt.ylabel('In-sample Accuracy/Loss Curve')
# plt.xlabel('Report step')
# #plt.axis([0,20,min(Minibatch_acc)*0.9,max(Minibatch_acc)*1.1])
# plt.legend()
# plt.show()