import math
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
# from xgboost import plot_importance
# from sklearn.ensemble import VotingClassifier

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
sns.set(font_scale=1)
color = sns.color_palette()


# Load Data

def load_data(data_path):

    with open(data_path + 'train_x.p', 'rb') as f:
        train_x = pickle.load(f)

    with open(data_path + 'train_y.p', 'rb') as f:
        train_y = pickle.load(f)

    with open(data_path + 'train_w.p', 'rb') as f:
        train_w = pickle.load(f)

    with open(data_path + 'valid_x.p', 'rb') as f:
        valid_x = pickle.load(f)

    with open(data_path + 'valid_y.p', 'rb') as f:
        valid_y = pickle.load(f)

    with open(data_path + 'valid_w.p', 'rb') as f:
        valid_w = pickle.load(f)

    return train_x, train_y, train_w, valid_x, valid_y, valid_w


# Logistic Regression

class LogisticRegression:

    importance = np.array([])
    indices = np.array([])

    def __init__(self, t_x, t_y, t_w):

        self.train_x = t_x
        self.train_y = t_y
        self.train_w = t_w

    def get_importance(self, clf):

        self.importance = np.abs(clf.coef_)[0]
        indices = np.argsort(self.importance)[::-1]

        feature_num = self.train_x.shape[1]

        for f in range(feature_num):
            print("%d. feature %d (%f)" % (f + 1, indices[f], self.importance[indices[f]]))

    def show(self):

        feature_num = self.train_x.shape[1]

        plt.figure(figsize=(20, 10))
        plt.title('Feature Importance in Logistic Regression')
        plt.bar(range(feature_num), self.importance[self.indices], color=color[0], align="center")
        plt.xticks(range(feature_num), self.indices)
        plt.xlim([-1, feature_num])
        plt.show()

    def train(self):

        clf_lr = LogisticRegression(random_state=1)
        '''
        LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                           penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
                           verbose=0, warm_start=False)
        '''

        clf_lr.fit(self.train_x, self.train_y, sample_weight=self.train_w)

        # K-fold cross-validation on Logistic Regression
        scores = cross_val_score(clf_lr, self.train_x, self.train_y, cv=10)
        print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))

        self.get_importance(clf_lr)


# k-Nearest Neighbor

class KNearestNeighbor:

    def __init__(self, t_x, t_y, t_w):

        self.train_x = t_x
        self.train_y = t_y
        self.train_w = t_w

    def train(self):

        clf_knn = KNeighborsClassifier()
        '''
        KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                             metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                             weights='uniform')
        '''

        clf_knn.fit(self.train_x, self.train_y)  # without parameter sample_weight

        # K-fold cross-validation on Logistic Regression
        scores = cross_val_score(clf_knn, self.train_x, self.train_y, cv=10)
        print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))


# SVM-SVC

class SupportVectorClustering:

    def __init__(self, t_x, t_y, t_w):

        self.train_x = t_x
        self.train_y = t_y
        self.train_w = t_w

    def train(self):

        clf_svc = SVC(random_state=1)

        clf_svc.fit(self.train_x, self.train_y, sample_weight=self.train_w)

        scores = cross_val_score(clf_svc, self.train_x, self.train_y, cv=10)
        print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))


# Gaussian NB

class GaussianNB:

    def __init__(self, t_x, t_y, t_w):

        self.train_x = t_x
        self.train_y = t_y
        self.train_w = t_w

    def train(self):

        clf_gnb = GaussianNB()

        clf_gnb.fit(self.train_x, self.train_y, sample_weight=self.train_w)

        scores = cross_val_score(clf_gnb, self.train_x, self.train_y, cv=10)
        print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))


# Decision Tree

class DecisionTree:

    importance = np.array([])
    indices = np.array([])

    def __init__(self, t_x, t_y, t_w):

        self.train_x = t_x
        self.train_y = t_y
        self.train_w = t_w

    def get_importance(self, clf):

        self.importance = clf.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = self.train_x.shape[1]

        for f in range(feature_num):
            print("%d. feature %d (%f)" % (f + 1, self.indices[f], self.importance[self.indices[f]]))

    def show(self):

        feature_num = self.train_x.shape[1]

        plt.figure(figsize=(20, 10))
        plt.title('Feature Importance in Decision Tree')
        plt.bar(range(feature_num), self.importance[self.indices], color=color[1], align="center")
        plt.xticks(range(feature_num), self.indices)
        plt.xlim([-1, feature_num])
        plt.show()

    def train(self):

        clf_dt = DecisionTreeClassifier(random_state=1)
        '''
        DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                               max_features=None, max_leaf_nodes=None,
                               min_impurity_decrease=0.0, min_impurity_split=None,
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, presort=False, random_state=1,
                               splitter='best')
        '''

        clf_dt.fit(self.train_x, self.train_y)

        scores = cross_val_score(clf_dt, self.train_x, self.train_y, cv=10)
        print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))

        self.get_importance(clf_dt)


# Random Forest

class RandomForest:

    importance = np.array([])
    indices = np.array([])
    std = np.array([])

    def __init__(self, t_x, t_y, t_w):

        self.train_x = t_x
        self.train_y = t_y
        self.train_w = t_w

    def get_importance(self, clf):

        self.importance = clf.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]
        self.std = np.std([clf.feature_importances_ for tree in clf.estimators_], axis=0)

        feature_num = self.train_x.shape[1]

        for f in range(feature_num):
            print("%d. feature %d (%f)" % (f + 1, self.indices[f], self.importance[self.indices[f]]))

    def show(self):

        feature_num = self.train_x.shape[1]

        plt.figure(figsize=(20, 10))
        plt.title('Feature Importance in Random Forest')
        plt.bar(range(feature_num), self.importance[self.indices],
                color=color[2], yerr=self.std[self.indices], align="center")
        plt.xticks(range(feature_num), self.indices)
        plt.xlim([-1, feature_num])
        plt.show()

    def train(self):

        clf_rf = RandomForestClassifier(random_state=1)
        '''
        RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                               max_depth=None, max_features='auto', max_leaf_nodes=None,
                               min_impurity_decrease=0.0, min_impurity_split=None,
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                               oob_score=False, random_state=1, verbose=0, warm_start=False)
        '''

        clf_rf.fit(self.train_x, self.train_y, self.train_w)

        scores = cross_val_score(clf_rf, self.train_x, self.train_y, cv=10)
        print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))

        self.get_importance(clf_rf)


# Extra Trees

class ExtraTrees:

    importance = np.array([])
    indices = np.array([])
    std = np.array([])

    def __init__(self, t_x, t_y, t_w):

        self.train_x = t_x
        self.train_y = t_y
        self.train_w = t_w

    def get_importance(self, clf):

        self.importance = clf.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]
        self.std = np.std([clf.feature_importances_ for tree in clf.estimators_], axis=0)

        feature_num = self.train_x.shape[1]

        for f in range(feature_num):
            print("%d. feature %d (%f)" % (f + 1, self.indices[f], self.importance[self.indices[f]]))

    def show(self):

        feature_num = self.train_x.shape[1]

        plt.figure(figsize=(20, 10))
        plt.title('Feature Importance in Extra Trees')
        plt.bar(range(feature_num), self.importance[self.indices],
                color=color[3], yerr=self.std[self.indices], align="center")
        plt.xticks(range(feature_num), self.indices)
        plt.xlim([-1, feature_num])
        plt.show()

    def train(self):

        clf_et = ExtraTreesClassifier(random_state=1)
        '''
        ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
                             max_depth=None, max_features='auto', max_leaf_nodes=None,
                             min_impurity_decrease=0.0, min_impurity_split=None,
                             min_samples_leaf=1, min_samples_split=2,
                             min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                             oob_score=False, random_state=1, verbose=0, warm_start=False)
        '''

        clf_et.fit(self.train_x, self.train_y, self.train_w)

        scores = cross_val_score(clf_et, self.train_x, self.train_y, cv=10)
        print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))

        self.get_importance(clf_et)


# AdaBoost

class AdaBoost:

    importance = np.array([])
    indices = np.array([])
    std = np.array([])

    def __init__(self, t_x, t_y, t_w):

        self.train_x = t_x
        self.train_y = t_y
        self.train_w = t_w

    def get_importance(self, clf):

        self.importance = clf.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]
        self.std = np.std([clf.feature_importances_ for tree in clf.estimators_], axis=0)

        feature_num = self.train_x.shape[1]

        for f in range(feature_num):
            print("%d. feature %d (%f)" % (f + 1, self.indices[f], self.importance[self.indices[f]]))

    def show(self):

        feature_num = self.train_x.shape[1]

        plt.figure(figsize=(20, 10))
        plt.title('Feature Importance in AdaBoost')
        plt.bar(range(feature_num), self.importance[self.indices],
                color=color[4], yerr=self.std[self.indices], align="center")
        plt.xticks(range(feature_num), self.indices)
        plt.xlim([-1, feature_num])
        plt.show()

    def train(self):

        clf_ab = AdaBoostClassifier(random_state=1)
        '''
        AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
                           learning_rate=1.0, n_estimators=50, random_state=1)
        '''

        clf_ab.fit(self.train_x, self.train_y, self.train_w)

        scores = cross_val_score(clf_ab, self.train_x, self.train_y, cv=10)
        print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))

        self.get_importance(clf_ab)


# GradientBoosting

class GradientBoosting:

    importance = np.array([])
    indices = np.array([])
    std = np.array([])

    def __init__(self, t_x, t_y, t_w):

        self.train_x = t_x
        self.train_y = t_y
        self.train_w = t_w

    def get_importance(self, clf):

        self.importance = clf.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]
        self.std = np.std([clf.feature_importances_ for tree in clf.estimators_], axis=0)

        feature_num = self.train_x.shape[1]

        for f in range(feature_num):
            print("%d. feature %d (%f)" % (f + 1, self.indices[f], self.importance[self.indices[f]]))

    def show(self):

        feature_num = self.train_x.shape[1]

        plt.figure(figsize=(20, 10))
        plt.title('Feature Importance in GradientBoosting')
        plt.bar(range(feature_num), self.importance[self.indices],
                color=color[5], yerr=self.std[self.indices], align="center")
        plt.xticks(range(feature_num), self.indices)
        plt.xlim([-1, feature_num])
        plt.show()

    def train(self):

        clf_gb = GradientBoostingClassifier(random_state=1)
        '''
        GradientBoostingClassifier(criterion='friedman_mse', init=None,
                                   learning_rate=0.1, loss='deviance', max_depth=3,
                                   max_features=None, max_leaf_nodes=None,
                                   min_impurity_decrease=0.0, min_impurity_split=None,
                                   min_samples_leaf=1, min_samples_split=2,
                                   min_weight_fraction_leaf=0.0, n_estimators=100,
                                   presort='auto', random_state=1, subsample=1.0, verbose=0,
                                   warm_start=False)
        '''

        clf_gb.fit(self.train_x, self.train_y, self.train_w)

        scores = cross_val_score(clf_gb, self.train_x, self.train_y, cv=10)
        print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))

        self.get_importance(clf_gb)


# XGBoost

class XGBoost:

    importance = np.array([])
    indices = np.array([])

    def __init__(self, t_x, t_y, t_w):

        self.train_x = t_x
        self.train_y = t_y
        self.train_w = t_w

    def get_importance(self, clf):

        self.importance = clf.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = self.train_x.shape[1]

        for f in range(feature_num):
            print("%d. feature %d (%f)" % (f + 1, self.indices[f], self.importance[self.indices[f]]))

    def show(self):

        feature_num = self.train_x.shape[1]

        plt.figure(figsize=(20, 10))
        plt.title('Feature Importance in XGBoost')
        plt.bar(range(feature_num), self.importance[self.indices],
                color=color[6], yerr=self.std[self.indices], align="center")
        plt.xticks(range(feature_num), self.indices)
        plt.xlim([-1, feature_num])
        plt.show()

    def train(self):
        clf_xgb = XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.8,
                                gamma=2, learning_rate=0.05, max_delta_step=0, max_depth=3,
                                min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
                                objective='binary:logistic', reg_alpha=0, reg_lambda=1,
                                scale_pos_weight=1, seed=0, silent=True, subsample=0.8)

        clf_xgb.fit(self.train_x, self.train_y, self.train_w)

        scores = cross_val_score(clf_xgb, self.train_x, self.train_y, cv=10)
        print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))

        self.get_importance(clf_xgb)


# Deep Neural Networks

class DeepNeuralNetworks:

    def __init__(self, t_x, t_y, t_w, v_x, v_y, v_w, hyper_para):

        # Inputs
        self.train_x = t_x
        self.train_y = t_y
        self.train_w = t_w
        self.valid_x = v_x
        self.valid_y = v_y
        self.valid_w = v_w

        # Hyperparameters
        self.version = hyper_para['version']
        self.epochs = hyper_para['epochs']
        self.layers_number = hyper_para['layers_number']
        self.unit_number = hyper_para['unit_number']
        self.learning_rate = hyper_para['learning_rate']
        self.keep_probability = hyper_para['keep_probability']
        self.batch_size = hyper_para['batch_size']
        self.display_step = hyper_para['display_step']
        self.save_path = hyper_para['save_path']
        self.log_path = hyper_para['log_path']

    # Input Tensors
    def input_tensor(self, n_feature):

        inputs_ = tf.placeholder(tf.float32, [None, n_feature], name='inputs')
        labels_ = tf.placeholder(tf.float32, None, name='labels')
        loss_weights_ = tf.placeholder(tf.float32, None, name='loss_weights')
        learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')
        keep_prob_ = tf.placeholder(tf.float32, name='keep_prob')
        is_train_ = tf.placeholder(tf.bool, name='is_train')

        return inputs_, labels_, loss_weights_, learning_rate_, keep_prob_, is_train_

    # Full Connected Layer
    def fc_layer(self, x_tensor, layer_name, num_outputs, keep_prob, training):

        with tf.name_scope(layer_name):
            x_shape = x_tensor.get_shape().as_list()

            weights = tf.Variable(tf.truncated_normal([x_shape[1], num_outputs], stddev=2.0 / math.sqrt(x_shape[1])))

            biases = tf.Variable(tf.zeros([num_outputs]))

            with tf.name_scope('fc_layer'):
                fc_layer = tf.add(tf.matmul(x_tensor, weights), biases)

                # Batch Normalization
                #  fc_layer = tf.layers.batch_normalization(fc_layer, training=training)

                # Activate function
                fc = tf.nn.relu(fc_layer)
                #  fc = tf.nn.elu(fc_layer)

            #  fc = tf.contrib.layers.fully_connected(x_tensor,
            #                                         num_outputs,
            #                                         weights_initializer=tf.truncated_normal_initializer(stddev=2.0 / math.sqrt(x_shape[1])),
            #                                         biases_initializer=tf.zeros_initializer())

            tf.summary.histogram('fc_layer', fc)

            fc = tf.nn.dropout(fc, keep_prob)

        return fc

    # Output Layer
    def output_layer(self, x_tensor, layer_name, num_outputs):

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

            out = tf.contrib.layers.fully_connected(x_tensor,
                                                    num_outputs,
                                                    activation_fn=None)

        return out

    # Model
    def model(self, x, n_layers, n_unit, keep_prob, is_training):

        #  fc1 = fc_layer(x, 'fc1', n_unit[0], keep_prob)
        #  fc2 = fc_layer(fc1, 'fc2', n_unit[1], keep_prob)
        #  fc3 = fc_layer(fc2, 'fc3', n_unit[2], keep_prob)
        #  fc4 = fc_layer(fc3, 'fc4', n_unit[3], keep_prob)
        #  fc5 = fc_layer(fc4, 'fc5', n_unit[4], keep_prob)

        #  logit_ = self.output_layer(fc5, 'output', 1)

        fc = []
        fc.append(x)
        for i in range(n_layers):
            fc.append(self.fc_layer(fc[i], 'fc{}'.format(i + 1), n_unit[i], keep_prob, is_training))

        logit_ = self.output_layer(fc[n_layers], 'output', 1)

        return logit_

    # LogLoss
    def log_loss(self, logit, weight, label):

        with tf.name_scope('prob'):
            prob = tf.nn.sigmoid(logit)

            #  with tf.name_scope('weight'):
            weight = weight / tf.reduce_sum(weight)

        with tf.name_scope('logloss'):
            #  loss = tf.losses.log_loss(labels=label, predictions=prob, weights=weight)
            loss = - tf.reduce_sum(weight * (label * tf.log(tf.clip_by_value(prob, 1e-10, 1.0)) + (1 - label) * tf.log(
                tf.clip_by_value((1 - prob), 1e-10, 1.0))))

        tf.summary.scalar('logloss', loss)

        return loss

    # Get Batches
    def get_batches(self, x, y, w, batch_num):

        n_batches = len(x) // batch_num

        for ii in range(0, n_batches * batch_num, batch_num):

            if ii != n_batches * batch_num:
                X, Y, W = x[ii: ii + batch_num], y[ii: ii + batch_num], w[ii: ii + batch_num]

            else:
                X, Y, W = x[ii:], y[ii:], w[ii:]

            yield X, Y, W

    # Training
    def train(self):

        # Build Network
        tf.reset_default_graph()
        train_graph = tf.Graph()

        with train_graph.as_default():

            # Inputs
            feature_num = list(self.train_x.shape)[1]
            inputs, labels, loss_weights, lr, keep_prob, is_train = self.input_tensor(feature_num)

            # Logits
            logits = self.model(inputs, self.layers_number, self.unit_number, keep_prob, is_train)
            logits = tf.identity(logits, name='logits')

            # Loss
            with tf.name_scope('Loss'):
                cost_ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
                #  cost_ = self.log_loss(logits, loss_weights, labels)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr).minimize(cost_)

            # LogLoss
            #  with tf.name_scope('LogLoss'):
            #      logloss = log_loss(logits, loss_weights, labels)

        # Training
        print('Training...')

        with tf.Session(graph=train_graph) as sess:

            # Merge all the summaries
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.log_path + self.version + '/train', sess.graph)
            valid_writer = tf.summary.FileWriter(self.log_path + self.version + '/valid')

            batch_counter = 0

            start_time = time.time()

            sess.run(tf.global_variables_initializer())

            for epoch_i in range(self.epochs):

                for batch_i, (batch_x, batch_y, batch_w) in enumerate(self.get_batches(self.train_x,
                                                                                       self.train_y,
                                                                                       self.train_w,
                                                                                       self.batch_size)):

                    batch_counter += 1

                    _, cost = sess.run([optimizer, cost_],
                                       {inputs: batch_x,
                                        labels: batch_y,
                                        loss_weights: batch_w,
                                        lr: self.learning_rate,
                                        keep_prob: self.keep_probability,
                                        is_train: True})

                    if batch_counter % self.display_step == 0 and batch_i > 0:

                        summary_train, cost_train = sess.run([merged, cost_],
                                                             {inputs: batch_x,
                                                              labels: batch_y,
                                                              loss_weights: batch_w,
                                                              keep_prob: 1.0,
                                                              is_train: False})
                        train_writer.add_summary(summary_train, batch_counter)

                        cost_valid_a = []

                        for iii, (valid_batch_x, valid_batch_y, valid_batch_w) in enumerate(self.get_batches(self.valid_x,
                                                                                                             self.valid_y,
                                                                                                             self.valid_w,
                                                                                                             self.batch_size)):
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

                        print("Epoch: {}/{} |".format(epoch_i + 1, self.epochs),
                              "Batch: {} |".format(batch_counter),
                              "Time: {:>3.2f}s |".format(total_time),
                              "Train_Loss: {:>.8f} |".format(cost_train),
                              "Valid_Loss: {:>.8f}".format(cost_valid))

            # Save Model
            print('Saving...')
            saver = tf.train.Saver()
            saver.save(sess, self.save_path + 'model.' + self.version + '.ckpt')


if __name__ == "__main__":

    # HyperParameters
    hyper_parameters = {'version': '1.0',
                        'epochs': 10,
                        'layers_number': 10,
                        'unit_number': [200, 400, 800, 800, 800, 800, 800, 800, 400, 200],
                        'learning_rate': 0.01,
                        'keep_probability': 0.75,
                        'batch_size': 512,
                        'display_step': 100,
                        'save_path': './checkpoints/',
                        'log_path': './log/'}

    pickled_data_path = './preprocessed_data/'

    print('Loading data set...')
    tr, tr_y, tr_w, val_x, val_y, val_w = load_data(pickled_data_path)

    dnn = DNN(tr, tr_y, tr_w, val_x, val_y, val_w, hyper_parameters)
    dnn.train()
    print('Done!')