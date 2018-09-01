from sklearn import datasets
import numpy as np
import tensorflow as tf

iris = datasets.load_iris()
num_labels = len(set(iris.target))
X = iris.data.astype(np.float32)
Y = (np.arange(num_labels) == np.array(iris.target)[:,None]).astype(np.float32)

#print(X.shape,y.shape)

alpha = 0.00001			#training rate
iters = 300000			#iterations

epoch = 6000

layers = [4, 16, 170, 50, 123, 15, 3]		#layers dimensions
L = len(layers)             #number of layers in the network

y = tf.placeholder(tf.float32, shape=[150,3], name = 'y-input')

#initalizing parameters thetas and biases in dictionary 'parameters'
parameters = {}
for l in range(1, L):
    parameters['theta' + str(l)] = tf.Variable(tf.random_uniform([layers[l-1],layers[l]], -1, 1), name = "theta" + str(l))
    parameters['bias' + str(l)] = tf.Variable(tf.zeros([layers[l]]), name = "bias"+str(l))

#definations of output for each layer
layer_out = dict()
layer_out['a1'] = tf.placeholder(tf.float32, shape=[150,4], name = 'x-input')
for i in range(2, L+1):
	layer_out['a'+str(i)] = tf.sigmoid(tf.matmul(layer_out['a' + str(i-1)], parameters['theta' + str(i-1)]) + parameters['bias'+str(i-1)])

#defining cost
cost = tf.reduce_mean(( (y * tf.log(layer_out['a'+ str(L)])) + ((1 - y) * tf.log(1.0 - layer_out['a'+ str(L)])) ) *(-1))

#update defination
train = tf.train.AdamOptimizer(alpha).minimize(cost)

#session initialization
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#running the model
for i in range(iters):
	sess.run(train, feed_dict={layer_out['a1']: X, y: Y})
	if i % epoch == 0:
		print('Epoch:', i)
		#print('Hypothesis ', sess.run(layer_out['a'+ str(L)], feed_dict={layer_out['a1']: X, y: Y}))
		print('cost ', sess.run(cost, feed_dict={layer_out['a1']: X, y: Y}))

print('Final Hypothesis ', sess.run(layer_out['a'+ str(L)], feed_dict={layer_out['a1']: X, y: Y}))
