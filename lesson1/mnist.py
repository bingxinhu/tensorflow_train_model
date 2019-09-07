import tensorflow as tf
#old_v = tf.compat.v1.logging.set_verbosity()
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
#tf.compat.v1.logging.set_verbosity(old_v)

inputSize  = 784
outputSize = 10
hiddenSize = 50
batchSize  = 64
trainCycle = 50000

# 输入层
inputLayer = tf.placeholder(tf.float32, shape=[None, inputSize])

# 隐藏层
hiddenWeight = tf.Variable(tf.truncated_normal([inputSize, hiddenSize], mean=0, stddev=0.1))
hiddenBias   = tf.Variable(tf.truncated_normal([hiddenSize]))
hiddenLayer  = tf.add(tf.matmul(inputLayer, hiddenWeight), hiddenBias)
hiddenLayer  = tf.nn.sigmoid(hiddenLayer)

# 输出层
outputWeight = tf.Variable(tf.truncated_normal([hiddenSize, outputSize], mean=0, stddev=0.1))
outputBias   = tf.Variable(tf.truncated_normal([outputSize], mean=0, stddev=0.1))
outputLayer  = tf.add(tf.matmul(hiddenLayer, outputWeight), outputBias)
outputLayer  = tf.nn.sigmoid(outputLayer)
 
# 标签
outputLabel = tf.placeholder(tf.float32, shape=[None, outputSize])
 
# 损失函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=outputLabel, logits=outputLayer))
 
# 优化器
optimizer = tf.train.AdamOptimizer()
 
# 训练目标
target = optimizer.minimize(loss)
 
# 训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
 
    for i in range(trainCycle):
        batch = mnist.train.next_batch(batchSize)
        sess.run(target, feed_dict={inputLayer: batch[0], outputLabel: batch[1]})
 
        if i % 1000 == 0:
            corrected = tf.equal(tf.argmax(outputLabel, 1), tf.argmax(outputLayer, 1))
            accuracy = tf.reduce_mean(tf.cast(corrected, tf.float32))
            accuracyValue = sess.run(accuracy, feed_dict={inputLayer: batch[0], outputLabel: batch[1]})
            print(i, 'train set accuracy:', accuracyValue)

# 测试
    corrected = tf.equal(tf.argmax(outputLabel, 1), tf.argmax(outputLayer, 1))
    accuracy  = tf.reduce_mean(tf.cast(corrected, tf.float32))
    accuracyValue = sess.run(accuracy, feed_dict={inputLayer: mnist.test.images, outputLabel: mnist.test.labels})
    print("accuracy on test set:", accuracyValue)

    sess.close()
