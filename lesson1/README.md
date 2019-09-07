1、获取MNIST数据

　　MNIST数据集只要一行代码就可以获取的到，非常方便。关于MNIST的基本信息可以参考我的上一篇随笔。

mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
 

2、模型基本结构

　　本次采用的训练模型为三层神经网络结构；

1 inputSize  = 784 # 定义输入为28x28图像，像素每一行的值为784
2 outputSize = 10  # 定义输出为10分类0 1 2 3 ...... 9
3 hiddenSize = 50  # 隐藏层节点数为50个
4 batchSize  = 64  # 每次训练mini-batch数量为64
5 trainCycle = 50000 #最大训练周期为50000
 

3、输入层

　　输入层用于接收每次小批量样本的输入，先通过placeholder来进行占位，在训练时才传入具体的数据。
   值得注意的是，在生成输入层的tensor时，传入的shape中有一个‘None’，表示每次输入的样本的数量，
   该‘None’表示先不作具体的指定，在真正输入的时候再根据实际的数据来进行推断。
   这个很方便，但也是有条件的，也就是通过该方法返回的tensor不能使用简单的加（+）减（-）乘（*）除（/）符号来进行计算（否则将会报错），
   需要用TensorFlow中的相关函数来进行代替。

inputLayer = tf.placeholder(tf.float32, shape=[None, inputSize])
 

4、隐藏层

　　在神经网络中，隐藏层的作用主要是提取数据的特征（feature）。
   f成，与上次采用的 tensorflow.
   random_normal() 不一样。这两者的作用都是生成指定形状、期望和标准差的符合正太分布随机变量。
   区别是 truncated_normal 函数对随机变量的范围有个限制（与期望的偏差在2个标准差之内，否则丢弃）。
   另外偏差项这里也使用了变量的形式，也可以采用常量来进行替代。 
　 激活函数为sigmoid函数。

1 hiddenWeight = tf.Variable(tf.truncated_normal([inputSize, hiddenSize], mean=0, stddev=0.1))
2 hiddenBias   = tf.Variable(tf.truncated_normal([hiddenSize]))
3 hiddenLayer  = tf.add(tf.matmul(inputLayer, hiddenWeight), hiddenBias)
4 hiddenLayer  = tf.nn.sigmoid(hiddenLayer)
 

5、输出层

　　输出层与隐藏层类似，只是节点数不一样。

1 outputWeight = tf.Variable(tf.truncated_normal([hiddenSize, outputSize], mean=0, stddev=0.1))
2 outputBias   = tf.Variable(tf.truncated_normal([outputSize], mean=0, stddev=0.1))
3 outputLayer  = tf.add(tf.matmul(hiddenLayer, outputWeight), outputBias)
4 outputLayer  = tf.nn.sigmoid(outputLayer)
 

6、输出标签

　　跟输入层一样，也是先占位，在最后训练的时候再传入具体的数据。标签，也就是每一个样本的正确分类。

outputLabel = tf.placeholder(tf.float32, shape=[None, outputSize])
 

7、损失函数

　　这里采用的是交叉熵损失函数。注意用的是v2版本，第一个版本已被TensorFlow声明为deprecated，准备废弃了。

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=outputLabel, logits=outputLayer))
 

8、优化器与目标函数

　　优化器采用了Adam梯度下降法，我试过了普通的GradientDescentOptimizer，效果不如Adam；也用过Adadelta，结果几乎收敛不了。

　　目标函数就是最小化损失函数。

optimizer = tf.train.AdamOptimizer()
target    = optimizer.minimize(loss)
 

9、训练过程

　　先创建一个会话，然后初始化tensors，最后进行迭代训练。模型的收敛速度很快，在1000次的时候就达到了大概90%的正确率。
 1 with tf.Session() as sess:
 2     sess.run(tf.global_variables_initializer())
 3 
 4     for i in range(trainCycle):
 5         batch = mnist.train.next_batch(batchSize)
 6         sess.run(target, feed_dict={inputLayer: batch[0], outputLabel: batch[1]})
 7 
 8         if i % 1000 == 0:
 9             corrected = tf.equal(tf.argmax(outputLabel, 1), tf.argmax(outputLayer, 1))
10             accuracy = tf.reduce_mean(tf.cast(corrected, tf.float32))
11             accuracyValue = sess.run(accuracy, feed_dict={inputLayer: batch[0], outputLabel: batch[1]})
12             print(i, 'train set accuracy:', accuracyValue)
----------------------------------------------------------------------------------------------------------
0 train set accuracy: 0.109375
1000 train set accuracy: 0.890625
2000 train set accuracy: 0.9375
3000 train set accuracy: 0.90625
4000 train set accuracy: 0.890625
5000 train set accuracy: 0.953125
6000 train set accuracy: 0.9375
7000 train set accuracy: 0.9375
8000 train set accuracy: 0.984375
9000 train set accuracy: 0.9375
10000 train set accuracy: 0.984375
11000 train set accuracy: 0.921875
12000 train set accuracy: 0.9375
13000 train set accuracy: 0.984375
14000 train set accuracy: 0.96875
15000 train set accuracy: 0.953125
16000 train set accuracy: 0.953125
17000 train set accuracy: 0.953125
18000 train set accuracy: 0.96875
19000 train set accuracy: 0.96875
20000 train set accuracy: 0.96875
21000 train set accuracy: 0.96875
22000 train set accuracy: 0.984375
23000 train set accuracy: 1.0
24000 train set accuracy: 0.984375
25000 train set accuracy: 0.984375
26000 train set accuracy: 0.96875
27000 train set accuracy: 0.984375
28000 train set accuracy: 0.96875
29000 train set accuracy: 0.984375
30000 train set accuracy: 1.0
31000 train set accuracy: 0.984375
32000 train set accuracy: 0.984375
33000 train set accuracy: 0.953125
34000 train set accuracy: 0.953125
35000 train set accuracy: 0.96875
36000 train set accuracy: 1.0
37000 train set accuracy: 1.0
38000 train set accuracy: 0.984375
39000 train set accuracy: 0.96875
40000 train set accuracy: 0.984375
41000 train set accuracy: 0.984375
42000 train set accuracy: 0.96875
43000 train set accuracy: 0.984375
44000 train set accuracy: 0.984375
45000 train set accuracy: 0.984375
46000 train set accuracy: 0.984375
47000 train set accuracy: 0.984375
48000 train set accuracy: 0.96875
49000 train set accuracy: 1.0
10、测试训练结果

　　在测数据集上测试。准确率达到96%，比单层的神经网络好很多。

   corrected = tf.equal(tf.argmax(outputLabel, 1), tf.argmax(outputLayer, 1))
   accuracy  = tf.reduce_mean(tf.cast(corrected, tf.float32))
   accuracyValue = sess.run(accuracy, feed_dict={inputLayer: mnist.test.images, outputLabel: mnist.test.labels})
   print("accuracy on test set:", accuracyValue)
   测试集上的输出：
  accuracy on test set: 0.9632
