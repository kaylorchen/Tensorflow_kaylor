import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist", one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples
print(n_batch)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# loss = tf.reduce_mean(tf.square(y-prediction))
loss = tf.compat.v1.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.2).minimize(loss)
init = tf.compat.v1.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

accuracy = tf.compat.v1.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.compat.v1.train.Saver()

print("start")

with tf.compat.v1.Session() as sess:
    sess.run(init)
    print(sess.run(accuracy, feed_dict={x: mnist.test.images[:1000], y:mnist.test.labels[:1000]}))
    saver.restore(sess,'net/my_net.ckpt')
    print(sess.run(accuracy, feed_dict={x: mnist.test.images[:1000], y: mnist.test.labels[:1000]}))