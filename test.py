import pickle
import math
import tensorflow as tf

version = '1.0'

load_path = './checkpoints/model.' + version + '.ckpt'


def sigmoid(x):
    for i in range(len(x)):
        x[i] = 1 / (1 + math.exp(-x[i]))
    return x


# Load Test Set

print('Loading test codes and labels...')

with open('test_x.p', 'rb') as f:
    test_f = pickle.load(f)

test_id = test_f[:, 0]
test_x = test_f[:, 1:89]

loaded_graph = tf.Graph()

with tf.Session(graph=loaded_graph) as sess:

        # Load Graph
        print('Loading model...')
        loader = tf.train.import_meta_graph(load_path + '.meta')
        loader.restore(sess, load_path)

        # Get Tensors
        inputs = loaded_graph.get_tensor_by_name('inputs:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        is_train = loaded_graph.get_tensor_by_name('is_train:0')
        logit = loaded_graph.get_tensor_by_name('logits:0')

        feed = {inputs: test_x,
                keep_prob: 1.0,
                is_train: False}

        # Get Probabilities of labels
        print('Getting prediction...')
        logits = sess.run(logit, feed_dict=feed)

        prob = sigmoid(logits)

        # Save data
        print('Saving csv...')
        with open('./results.csv', 'w') as f:
            res_str = 'id,proba' + '\n'
            f.writelines(res_str)
            for i in range(len(prob)):
                res_str = str(int(test_id[i])) + ',' + str(float(prob[i])) + '\n'
                f.writelines(res_str)

print('Done!')
