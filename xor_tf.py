#XOR gate in python using tensorflow
import tensorflow as tf

#input matrices
X = [[0,0],[0,1],[1,0],[1,1]]
Y = [[0],[1],[1],[0]]

#initializing hyperparameters
alpha = 0.03
hidden_layer = 8
iters = 100000

epoch = 10000

#Phase 1: assembling the graph

#creating placeholders for input and labels
a1 = tf.placeholder(tf.float32, shape=[4,2], name = 'x-input')
y = tf.placeholder(tf.float32, shape=[4,1], name = 'y-input')

#creating weight and bias
theta1 = tf.Variable(tf.random_uniform([2,hidden_layer], -1, 1), name = "theta1")
theta2 = tf.Variable(tf.random_uniform([hidden_layer,1], -1, 1), name = "theta2")

bias1 = tf.Variable(tf.zeros([hidden_layer]), name = "bias1")
bias2 = tf.Variable(tf.zeros([1]), name = "bias2")

#model to predict y
a2         = tf.sigmoid(tf.matmul(a1, theta1) + bias1)
hypothesis = tf.sigmoid(tf.matmul(a2, theta2) + bias2)

#cost definition
cost = tf.reduce_mean(( (y * tf.log(hypothesis)) + ((1 - y) * tf.log(1.0 - hypothesis)) ) *(-1))

#optimizing weights and biases
train = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

#Phase 2: executing the graph

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#training the model
for i in range(iters):
	sess.run(train, feed_dict={a1: X, y: Y})
	if i % epoch == 0:
		print('Epoch ', i)
		print('Hypothesis ', sess.run(hypothesis, feed_dict={a1: X, y: Y}))
		print('cost ', sess.run(cost, feed_dict={a1: X, y: Y}))

#Final Hypothesis
print('Final Hypothesis ', sess.run(hypothesis, feed_dict={a1: X, y: Y}))
