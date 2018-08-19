import tensorflow as tf
from tensorflow.contrib.layers import flatten

def model(x, keep_prob):
    mu = 0
    sigma = 0.1

    # 1x1x3 conv (in: 32x32x3, out: 32x32x3)
    conv1_W = tf.Variable(tf.truncated_normal(shape=(1, 1, 3, 3), mean = mu, stddev = sigma), name="con1_W")
    conv1_b = tf.Variable(tf.zeros(3), name="conv1_b")
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID')
    x = tf.nn.bias_add(conv1, conv1_b)

    # 5x5x6 conv (in: 32x32x3, out: 28x28x6)
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma), name="conv2_W")
    conv2_b = tf.Variable(tf.zeros(6), name="conv2_b")
    conv2 = tf.nn.conv2d(x, conv2_W, strides=[1, 1, 1, 1], padding='VALID')
    x = tf.nn.bias_add(conv2, conv2_b)

    # relu + 2x2 max pooling (in: 28x28x6, out: 14x14x6)
    x = tf.nn.relu(x)
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # 5x5x16 conv (in: 14x14x6, out: 10x10x16)
    conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma), name="conv3_W")
    conv3_b = tf.Variable(tf.zeros(16), name="conv3_b")
    conv3 = tf.nn.conv2d(x, conv3_W, strides=[1, 1, 1, 1], padding='VALID')
    x = tf.nn.bias_add(conv3, conv3_b)

    # relu + 2x2 max pooling (in: 10x10x16, out: 5x5x16)
    x = tf.nn.relu(x)
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # make a copy for later concatenation
    x1 = x

    # 5x5x400 conv (in: 5x5x16, out: 1x1x400)
    conv4_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 400), mean = mu, stddev = sigma), name="conv4_W")
    conv4_b = tf.Variable(tf.zeros(400), name="conv4_b")
    conv4 = tf.nn.conv2d(x, conv4_W, strides=[1, 1, 1, 1], padding='VALID')
    x = tf.nn.bias_add(conv4, conv4_b)

    # relu
    x = tf.nn.relu(x)

    # flatten x1 (in: 5x5x16, out: 400)
    x1_flat = flatten(x1)

    # flatten x (in: 1x1x400, out: 400)
    x_flat = flatten(x)

    # concat x and x1 (in: 400 + 400, out: 800)
    x = tf.concat([x_flat, x1_flat], 1)

    # dropout
    fc0 = tf.nn.dropout(x, keep_prob)

    # fc layer (in: 800, out: 120)
    fc1_W = tf.Variable(tf.truncated_normal(shape=(800, 120), mean = mu, stddev = sigma), name="fc1_W")
    fc1_b = tf.Variable(tf.zeros(120), name="fc1_b")
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # relu + dropout
    fc1    = tf.nn.relu(fc1)
    fc1    = tf.nn.dropout(fc1, keep_prob)

    # fc layer (in: 120, out: 43)
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 43), mean = mu, stddev = sigma), name="fc2_W")
    fc2_b  = tf.Variable(tf.zeros(43), name="fc2_b")
    logits    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # regularization
    regularizer = tf.nn.l2_loss(conv1_W) + tf.nn.l2_loss(conv2_W) + tf.nn.l2_loss(conv3_W) + tf.nn.l2_loss(conv4_W) + tf.nn.l2_loss(fc1_W) + tf.nn.l2_loss(fc2_W)
    
    return logits, regularizer
