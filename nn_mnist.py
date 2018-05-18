import gzip
import pickle as cPickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
f.close()

train_x, train_y = train_set
train_y = one_hot(train_y, 10)

valid_x, valid_y = valid_set
valid_y = one_hot(valid_y, 10)

test_x, test_y = test_set
test_y = one_hot(test_y, 10)

x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 20)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(20)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(20, 10)) * 0.1)  #### You should try to quit the *0.1 and execute it
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

opt = tf.train.GradientDescentOptimizer(0.01)
# train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01
train = opt.minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

batch_size = 100
errors = []
validation_errors = []
dif = 100
epoch = 0
lastError = 0

while dif > 0.003:
    # train
    for jj in range(int(len(train_x) / batch_size)):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    # Get validation error
    validation_error = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})/(len(valid_y))
    validation_errors.append(validation_error)

    # # Print each Validation error
    # print("Epoch #:", ++epoch)
    # print("Validation Error: ", validation_error)
    # result = sess.run(y, feed_dict={x: valid_x})
    # for b, r in zip(batch_ys, result):
    #     print("     ", r, "-->", b)
    # print("--------------------------------")

    # Updating difference
    dif = abs(validation_error - lastError)
    lastError = validation_error
    print(dif)

# Get the error using test dataset
test_errors = sess.run(y, feed_dict={x: test_x, y_: test_y})

fails = 0
hits = 0

for b, r in zip(test_y, test_errors):
    if np.argmax(b) == np.argmax(r):
        hits += 1
    else:
        fails += 1
    # print(b, "-->", r)

print("Hits: ", hits)
print("Fails: ", fails)
print("Hits Percentage: ", 100 * float(hits) / float(hits+fails), "%")
print("----------------------------------------------------------------------------------")


# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.pyplot as plt

# plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
# plt.show()  # Let's see a sample

print("Validation errors", validation_errors)
plt.plot(errors)
plt.plot(validation_errors)
plt.show()
