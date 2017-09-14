import math
import time
import pickle
import tensorflow as tf


# Hyperparameters

epochs = 100
layers_number = 10
unit_number = [200, 400, 800, 800, 800, 800, 800, 800, 400, 200]
learning_rate = 0.01
keep_probability = 0.75
batch_size = 512
display_step = 10

version = '1.0'
save_path = './saver/'


# Load Data

with open('train_x.p', 'rb') as f:
    train_x = pickle.load(f)

with open('train_y.p', 'rb') as f:
    train_y = pickle.load(f)

with open('train_w.p', 'rb') as f:
    train_w = pickle.load(f)

with open('valid_x.p', 'rb') as f:
    valid_x = pickle.load(f)

with open('valid_y.p', 'rb') as f:
    valid_y = pickle.load(f)

with open('valid_w.p', 'rb') as f:
    valid_w = pickle.load(f)


# Get Batches
def get_batches(x, y, w, batch_num):

    n_batches = len(x) // batch_num

    for ii in range(0, n_batches*batch_num, batch_num):

        if ii != n_batches * batch_num:
            X, Y, W = x[ii: ii + batch_num], y[ii: ii + batch_num], w[ii: ii + batch_num]

        else:
            X, Y, W = x[ii:], y[ii:], w[ii:]

        yield X, Y, W


# Inputs

def input_tensor(n_feature):

    inputs_ = tf.placeholder(tf.float32, [None, n_feature], name='inputs')
    labels_ = tf.placeholder(tf.float32, None, name='labels')
    loss_weights_ = tf.placeholder(tf.float32, None, name='loss_weights')
    learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob_ = tf.placeholder(tf.float32, name='keep_prob')
    is_train_ = tf.placeholder(tf.bool)

    return inputs_, labels_, loss_weights_, learning_rate_, keep_prob_, is_train_


# Full Connected Layer

def fc_layer(x_tensor, layer_name, num_outputs, keep_prob, training):

    with tf.name_scope(layer_name):
        x_shape = x_tensor.get_shape().as_list()

        weights = tf.Variable(tf.truncated_normal([x_shape[1], num_outputs], stddev=2.0 / math.sqrt(x_shape[1])))

        biases = tf.Variable(tf.zeros([num_outputs]))

        with tf.name_scope('fc_layer'):
            fc_layer = tf.add(tf.matmul(x_tensor, weights), biases)
            fc = tf.nn.relu(fc_layer)
        tf.summary.histogram('fc_layer', fc)

        #  fc = tf.contrib.layers.fully_connected(x_tensor,
        #                                         num_outputs,
        #                                         weights_initializer=tf.truncated_normal_initializer(stddev=2.0 / math.sqrt(x_shape[1])),
        #                                         biases_initializer=tf.zeros_initializer())
        #
        #  tf.summary.histogram('fc_layer', fc)

        # Batch Normalization
        fc = tf.layers.batch_normalization(fc, training=training)

        fc = tf.nn.dropout(fc, keep_prob)

    return fc


# Output Layer

def output_layer(x_tensor, layer_name, num_outputs):

    with tf.name_scope(layer_name):

        #  x_shape = x_tensor.get_shape().as_list()
        #
        #  weights = tf.Variable(tf.truncated_normal([x_shape[1], num_outputs], stddev=2.0 / math.sqrt(x_shape[1])))
        #
        #  biases = tf.Variable(tf.zeros([num_outputs]))
        #
        #  with tf.name_scope('Wx_plus_b'):
        #      output_layer = tf.add(tf.matmul(x_tensor, weights), biases)
        #  tf.summary.histogram('output', output_layer)

        output_layer = tf.contrib.layers.fully_connected(x_tensor,
                                                         num_outputs,
                                                         activation_fn=None)

    return output_layer


# Model

def model(x, n_layers, n_unit, keep_prob, is_training):

    #  fc1 = fc_layer(x, 'fc1', n_unit[0], keep_prob)
    #  fc2 = fc_layer(fc1, 'fc2', n_unit[1], keep_prob)
    #  fc3 = fc_layer(fc2, 'fc3', n_unit[2], keep_prob)
    #  fc4 = fc_layer(fc3, 'fc4', n_unit[3], keep_prob)
    #  fc5 = fc_layer(fc4, 'fc5', n_unit[4], keep_prob)

    #  logit_ = output_layer(fc5, 'output', 1)

    fc = []
    fc.append(x)
    for i in range(n_layers):
        fc.append(fc_layer(fc[i], 'fc{}'.format(i+1), n_unit[i], keep_prob, is_training))

    logit_ = output_layer(fc[n_layers], 'output', 1)

    return logit_


# LogLoss

def log_loss(logit, weight, label):

    with tf.name_scope('prob'):
        prob = tf.nn.sigmoid(logit) / 2.0 + 1.0

    with tf.name_scope('weight'):
        weight = weight / tf.reduce_sum(weight)

    with tf.name_scope('logloss'):
        #  loss = tf.losses.log_loss(labels=label, predictions=prob, weights=weight)
        loss = - tf.reduce_sum(weight * (label * tf.log(tf.clip_by_value(prob, 1e-10, 1.0)) + (1 - label) * tf.log(tf.clip_by_value((1 - prob), 1e-10, 1.0))))

    tf.summary.scalar('logloss', loss)

    return loss


# Build Network

tf.reset_default_graph()

train_graph = tf.Graph()

with train_graph.as_default():

    # Inputs
    feature_num = list(train_x.shape)[1]
    inputs, labels, loss_weights, lr, keep_prob, is_train = input_tensor(feature_num)

    # Logits
    logits = model(inputs, layers_number, unit_number, keep_prob, is_train)
    logits = tf.identity(logits, name='logits')

    # Loss
    with tf.name_scope('Loss'):
        cost_ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
        #  cost_ = log_loss(logits, loss_weights, labels)

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr).minimize(cost_)

    # LogLoss
    #  with tf.name_scope('LogLoss'):
    #      logloss = log_loss(logits, loss_weights, labels)


# Tarining

print('Training...')

with tf.Session(graph=train_graph) as sess:

    # Merge all the summaries
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./log/' + version + '/train', sess.graph)
    valid_writer = tf.summary.FileWriter('./log/' + version + '/valid')

    batch_counter = 0

    start_time = time.time()

    sess.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):

        for batch_i, (batch_x, batch_y, batch_w) in enumerate(get_batches(train_x,
                                                                          train_y,
                                                                          train_w,
                                                                          batch_size)):

            batch_counter += 1

            _, cost = sess.run([optimizer, cost_],
                               {inputs: batch_x,
                                labels: batch_y,
                                loss_weights: batch_w,
                                lr: learning_rate,
                                keep_prob: keep_probability,
                                is_train: True})

            if batch_counter % display_step == 0 and batch_i > 0:

                summary_train, cost_train = sess.run([merged, cost_],
                                                     {inputs: batch_x,
                                                      labels: batch_y,
                                                      loss_weights: batch_w,
                                                      keep_prob: 1.0,
                                                      is_train: False})
                train_writer.add_summary(summary_train, batch_counter)

                cost_valid_a = []

                for iii, (valid_batch_x, valid_batch_y, valid_batch_w) in enumerate(get_batches(valid_x,
                                                                                                valid_y,
                                                                                                valid_w,
                                                                                                batch_size)):

                    summary_valid_i, cost_valid_i = sess.run([merged, cost_],
                                                             {inputs: valid_batch_x,
                                                              labels: valid_batch_y,
                                                              loss_weights: valid_batch_w,
                                                              keep_prob: 1.0,
                                                              is_train: False})

                    cost_valid_a.append(cost_valid_i)

                cost_valid = sum(cost_valid_a) / len(cost_valid_a)

                valid_writer.add_summary(summary_valid_i, batch_counter)

                end_time = time.time()
                total_time = end_time - start_time

                print("Epoch: {}/{} |".format(epoch_i+1, epochs),
                      "Batch: {} |".format(batch_counter),
                      "Time: {:>3.2f}s |".format(total_time),
                      "Train_Loss: {:>.8f} |".format(cost_train),
                      "Valid_Loss: {:>.8f}".format(cost_valid))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_path)
