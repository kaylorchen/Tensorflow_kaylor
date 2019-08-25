import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

n_input = 28 #输入一行，一行有28个数据
max_time = 28 #一共28行
lstm_size = 100 # 隐藏层单元数
n_classes = 10 # 10个分类
batch_size = 50 # 每批次50个样本
n_batch = mnist.train.num_examples // batch_size #计算一共有多少个批次

#这里的None是表示第一个维度是可以任意长度
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32,[None,784],name='input_x')
    y = tf.placeholder(tf.float32,[None, 10],name='input_y')

weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))

biases = tf.Variable(tf.constant(0.1,shape=[n_classes]))

def RNN(X, weights, biases):
    inputs =  tf.reshape(X, [-1, max_time, n_input])
    lstm_cell =  tf.nn.rnn_cell.BasicLSTMCell(lstm_size)

    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], weights) + biases)
    return  results

prediction =  RNN(x, weights, biases)

cross_entropy =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(6):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs, y:batch_ys})
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images[:1000], y:mnist.test.labels[:1000]})
        print("Iter" + str(epoch) + ", testing accuracy: " + str(acc))

