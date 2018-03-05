#### Generalized version of XOR gate neural network in python using tensorflow ####

## There is 'layers' list in which you can add as many layers you want(as much as your system handles) 
## Each integer in 'layers' list represents number of neurons in that layer 
## Therefore first layer being input layer has 2 neurons incase of simple xor gate and last layer, output layer has 1 neuron incase of one output 

import tensorflow as tf

#input matrices
X = [[0,0],[0,1],[1,0],[1,1]]
Y = [[0],[1],[1],[0]]

alpha = 0.03            #training rate
iters = 100000          #iterations

epoch = 10000

layers = [2, 6,4 , 1]       #layers dimensions
L = len(layers)             #number of layers in the network

y = tf.placeholder(tf.float32, shape=[4,1], name = 'y-input')

#initalizing parameters thetas and biases in dictionary 'parameters'
parameters = {}
for l in range(1, L):
    parameters['theta' + str(l)] = tf.Variable(tf.random_uniform([layers[l-1],layers[l]], -1, 1), name = "theta" + str(l))
    parameters['bias' + str(l)] = tf.Variable(tf.zeros([layers[l]]), name = "bias"+str(l))

#definations of output for each layer
layer_out = dict()
layer_out['a1'] = tf.placeholder(tf.float32, shape=[4,2], name = 'x-input')
for i in range(2, L+1):
    layer_out['a'+str(i)] = tf.sigmoid(tf.matmul(layer_out['a' + str(i-1)], parameters['theta' + str(i-1)]) + parameters['bias'+str(i-1)])

#defining cost
cost = tf.reduce_mean(( (y * tf.log(layer_out['a'+ str(L)])) + ((1 - y) * tf.log(1.0 - layer_out['a'+ str(L)])) ) *(-1))

#update defination
train = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

#session initialization
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#running the model
for i in range(iters):
    sess.run(train, feed_dict={layer_out['a1']: X, y: Y})
    if i % epoch == 0:
        print('Epoch:', i)
        print('Hypothesis ', sess.run(layer_out['a'+ str(L)], feed_dict={layer_out['a1']: X, y: Y}))
        print('cost ', sess.run(cost, feed_dict={layer_out['a1']: X, y: Y}))

print('Final Hypothesis ', sess.run(layer_out['a'+ str(L)], feed_dict={layer_out['a1']: X, y: Y}))
