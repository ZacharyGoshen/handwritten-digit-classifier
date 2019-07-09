from read_image_data import retrieve_image_data
from read_labels import retrieve_labels
import numpy as np
import tensorflow as tf

# Retrieve the training data
training_images = retrieve_image_data("./../data/train-images.idx3-ubyte", range(0, 60000))
training_labels = retrieve_labels("./../data/train-labels.idx1-ubyte", range(0, 60000))

# Retrieve the test data
test_images = retrieve_image_data("./../data/t10k-images.idx3-ubyte", range(0, 10000))
test_labels = retrieve_labels("./../data/t10k-labels.idx1-ubyte", range(0, 10000))

# Declare hyperparameters
learning_rate = .0005
lamb = 0.01
epochs = 10
batch_size = 100

# Declare the training data placeholders
x = tf.compat.v1.placeholder(tf.float32, [None, 784]) # 28 * 28 = 784 pixels
y = tf.compat.v1.placeholder(tf.float32, [None, 10]) # 10 digits

# Reshape the input data to prepare for convolution
x_reshaped = tf.reshape(x, [-1, 28, 28, 1])

def create_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
	"""
	Return a new convolutional layer called "name" that is applied to "input_data". Specify depth of 
	the input and output with "num_input_channels" and "num_filters". Also specify the shape of the
	convolutional filters and max pooling dimensions with "filter_shape" and "pool_shape"
	"""

	# Set up the shape of the convolutional filters
	conv_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

	# Initialize weights and bias for the convolutional filters
	W = tf.Variable(tf.random.normal(conv_shape, stddev = 0.03), name = name + '_W')
	b = tf.Variable(tf.random.normal([num_filters]), name = name + '_b')

	# Set up the convolutional layer operation and add the bias
	output_layer = tf.nn.conv2d(input_data, W, [1, 1, 1, 1], padding = "SAME") + b

	# Apply ReLU activation function
	output_layer = tf.nn.relu(output_layer)

	# Perform 2 x 2 max pooling
	ksize_and_strides = [1, pool_shape[0], pool_shape[1], 1] # no overlap so dimensions and strides are equal
	output_layer = tf.nn.max_pool2d(output_layer, ksize = ksize_and_strides, strides = ksize_and_strides, padding = "SAME")

	return output_layer

# Create new convolutional layers
conv_layer1 = create_conv_layer(x_reshaped, 1, 32, [5, 5], [2, 2], "layer1")
conv_layer2 = create_conv_layer(conv_layer1, 32, 64, [5, 5], [2, 2], "layer2")

# Flatten the output of the convolutional layers to a 1d array
x_flattened = tf.reshape(conv_layer2, [-1, 7 * 7 * 64])

# Declare weights connecting last convolutional layer to dense layer
W1 = tf.Variable(tf.random.normal([7 * 7 * 64, 1000], stddev = 0.03), name = "W1")
b1 = tf.Variable(tf.random.normal([1000]), name = "b1")

# Declare weights connecting dense layer to output layer
W2 = tf.Variable(tf.random.normal([1000, 10], stddev = 0.03), name = "W2")
b2 = tf.Variable(tf.random.normal([10]), name = "b2")

# Calculate output of dense layer
dense_output = tf.add(tf.matmul(x_flattened, W1), b1)
dense_output = tf.nn.relu(dense_output)

# Calculate output of the final layer and clip it to avoid values less than zero or greater than one
y_pred = tf.nn.softmax(tf.add(tf.matmul(dense_output, W2), b2))
y_pred = tf.clip_by_value(y_pred, 1e-10, 0.9999999)

# Define cost function using cross entropy function an l2 regularization
cross_entropy = - tf.reduce_mean(tf.reduce_sum(y * tf.math.log(y_pred) + (1 - y) * tf.math.log(1 - y_pred), axis = 1))
l2_regularization = (lamb / 2) * (tf.reduce_mean(tf.reduce_sum(tf.math.square(W1), axis = 1)) + tf.reduce_mean(tf.reduce_sum(tf.math.square(W2), axis = 1)))
cost_function = cross_entropy + l2_regularization

# Create optimizer
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost_function)

# Setup the initialization operator
init = tf.compat.v1.global_variables_initializer()

# Define an accuracy assessment function
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def next_batch(batch_size, x, y):
	'''
	Return "batch_size" random samples and labels
	'''

	indices = np.arange(0, len(x))
	np.random.shuffle(indices)
	indices = indices[:batch_size]
	batch_x = [x[index] for index in indices]
	batch_y = [y[index] for index in indices]

	return np.asarray(batch_x), np.asarray(batch_y)

# Start the session
with tf.compat.v1.Session() as sess:
	# Initialize the variables
	sess.run(init)

	# Calculate number of batches to be run
	num_batches = int(len(training_images) / batch_size)

	for epoch in range(epochs):
		avg_cost = 0
		for i in range(num_batches):
			batch_x, batch_y = next_batch(batch_size, training_images, training_labels)
			_, cost = sess.run([optimizer, cost_function], feed_dict = {x : batch_x, y : batch_y})
			avg_cost += (cost / num_batches)
		print("Epoch: %d cost = %1.3f" % (epoch + 1, avg_cost))

	# Run test data through model
	print(sess.run(accuracy, feed_dict = {x : test_images, y : test_labels}))