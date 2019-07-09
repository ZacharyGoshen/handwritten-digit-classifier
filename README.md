# MNIST Handwritten Digit Classifier

The MNIST data set is a library of 60,000 images of handwritten digits. Each image is a 28 by 28 grayscale image of a digit 0 through 9. The data set is split into a training set of 50,000 images (5000 of each class) and a test set of 10,000 images (1000 of each class). *classifier.py* uses low level TensorFlow to create a fiver layer convolutional neural network that is able to correctly classify the test images with 97-98% accuracy.

The CNN has two convolutional layers and a depth layer. Both convolutional layers use 5 by 5 filters and 2 by 2 max pooling with no overlap. The first layer has one channel and 32 filters and the second layer has 32 channels and 64 filters. The depth layer contains 1000 nodes. The cost function uses cross entropy and L2 regularization. The network is trained using mini batch gradient descent with batches of size 100.

The image data is contained in binary files. *read_image_data.py* converts the binary image data to NumPy matrices while also flattening and standardizing it. *read_labels.py* converts the binary label data to a NumPy matrix of one hot encodings.

Running *classifier.py* will preprocess the data and build, train, and test the model. Training the model takes about 15 minutes using a standard performance GPU.

MNIST data set was taken from http://yann.lecun.com/exdb/mnist/
