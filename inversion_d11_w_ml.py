from __future__ import print_function
import numpy as np
import tensorflow as tf

from tensorflow.python.ops import resources
from tensorflow.contrib.tensor_forest.python import tensor_forest

X_train, X_validation, Y_train, Y_validation=np.load('seis_inversion.npy')
X_train, X_validation, Y_train, Y_validation=np.float32(X_train), np.float32(X_validation), \
                                        np.float32(Y_train), np.float32(Y_validation)
n_samples,num_features=X_train.shape
num_classes=Y_train.shape[1]
# batch_size=n_samples
# batch_size,num_features,num_classes

# number of features 
num_features = num_features
# number of target labels 
num_labels = num_classes
# learning rate (alpha) 
learning_rate = 0.05
# batch size 
batch_size = 128
# number of epochs 
num_steps = 5001

# input data 
train_dataset = X_train
train_labels = Y_train
# test_dataset = mnist.test.images 
# test_labels = mnist.test.labels 
valid_dataset = X_validation
valid_labels = Y_validation

# initialize a tensorflow graph 
graph = tf.Graph() 

with graph.as_default(): 
	""" 
	defining all the nodes 
	"""

	# Inputs 
	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, num_features)) 
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels)) 
# 	tf_valid_dataset = tf.constant(valid_dataset) 
# 	tf_test_dataset = tf.constant(test_dataset) 

	# Variables. 
	weights = tf.Variable(tf.truncated_normal([num_features, num_labels])) 
	biases = tf.Variable(tf.zeros([num_labels])) 
    
#     optcost = tf.reduce_sum(tf.pow(pred1-Y, 2))/(2*n_samples)
# # Gradient descent
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(optcost)

	# Training computation. 
	logits = tf.matmul(tf_train_dataset, weights) + biases 
	loss = tf.reduce_mean(tf.pow(logits-tf_train_labels,2)) 

	# Optimizer. 
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss) 

	# Predictions for the training, validation, and test data. 
	train_prediction = logits
# 	valid_prediction = tf.matmul(tf_valid_dataset, weights) + biases
# 	test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases) 


# utility function to calculate accuracy 
# def accuracy(predictions, labels): 
# 	correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) 
# 	accu = (100.0 * correctly_predicted) / predictions.shape[0] 
# 	return accu 

with tf.Session(graph=graph) as session: 
	# initialize weights and biases 
	tf.global_variables_initializer().run() 
	print("Initialized") 

	for step in range(num_steps): 
		# pick a randomized offset 
		offset = np.random.randint(0, train_labels.shape[0] - batch_size - 1) 

		# Generate a minibatch. 
		batch_data = train_dataset[offset:(offset + batch_size), :] 
		batch_labels = train_labels[offset:(offset + batch_size), :] 

		# Prepare the feed dict 
		feed_dict = {tf_train_dataset : batch_data, 
					tf_train_labels : batch_labels} 

		# run one step of computation 
		_, l, predictions = session.run([optimizer, loss, train_prediction], 
										feed_dict=feed_dict) 

		if (step % 500 == 0): 
			print("Minibatch loss at step {0}: {1}".format(step, l)) 
# 			print("Minibatch accuracy: {:.1f}%".format( 
# 				accuracy(predictions, batch_labels))) 
# 			print("Validation accuracy: {:.1f}%".format( 
# 				accuracy(valid_prediction.eval(), valid_labels))) 

# 	print("\nTest accuracy: {:.1f}%".format( 
# 		accuracy(test_prediction.eval(), test_labels))) 
