import sys
import os
import Queue
import numpy as np
import tensorflow as tf
import threading
import time


class Model(object):
    def __init__(self, data_dir, learning_rate=0.01, inf_lr=0.5,
                 feature_dim=1836, label_dim=159, weight_decay=0, num_hidden=150,
                 num_pairwise=16, include_second_layer=False, num_indicators=3,
                 inner_iter=15, inf_momentum=0.9):
        """
        Parameters
        ----------
        data_dir : basestring
            where to store the model and logs for tensorboard
        learning_rate : float
            learning rate for training the model
        inf_lr : float
            learning rate for the inference optimizer
        feature_dim : int
            dimensionality of the input features
        label_dim : int
            dimensionality of the output labels
        weight_decay : float
            the weight decay
        num_hidden : int
            number of hidden units for the linear part
        num_pairwise : int
            number of pairwise units for the global (label interactions) part
        include_second_layer : bool
            include a linear layer after the two-layer perceptron (as done in SPENs)
        num_indicators : int
            number of indicators used for cardinality range prediction
        inner_iter: int
            iterations of the inner (inference) optimization
        inf_momentum: float
            momentum used for the inner (inference) optimization
        """
        self.sess = tf.InteractiveSession()
        self.sentinel = object()
        self.feature_dim = feature_dim
        self.label_dim = label_dim
        self.learning_rate = learning_rate
        self.current_step = 0
        self.inf_lr = inf_lr

        self.num_hidden = num_hidden
        self.num_pairwise = num_pairwise
        self.include_second_layer = include_second_layer
        self.inf_r = 2  # The number of iterations used for Dykstra's algorithm

        self.weight_decay = weight_decay
        self.regularizer = tf.nn.l2_loss
        self.nonlinearity = tf.nn.relu
        self.num_indicators = num_indicators
        self.inner_iter = inner_iter
        self.inf_momentum = inf_momentum
        self.build_graph()

        # Create a summary writer
        self.data_dir = data_dir
        self.writer = tf.summary.FileWriter('%s/log/train' % self.data_dir)
        self.val_writer = tf.summary.FileWriter('%s/log/val' % self.data_dir)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)

        # Initialize variables
        tf.global_variables_initializer().run()

    def restore(self, path):
        """restore weights at `path`"""
        self.saver.restore(self.sess, path)
        self.mean = np.load("%s/mean.npz.npy" % os.path.dirname(path))
        self.std = np.load("%s/std.npz.npy" % os.path.dirname(path))

    def train_model(self, train_features, train_labels, epochs=1000, batch_size=50, train_ratio=0.9):
        """Train the model.

        Parameters
        ----------
        train_features
            features used for training/validation
        train_labels
            the corresponding labels
        epochs : int
            the number of epochs to train for
        batch_size : int
            batch size to use
        train_ratio : float
            defines the ratio of the data used for training. The rest is used for validation

        Returns
        -------
        f1 score on validation at the end of training

        """
        f1_scores = None
        train_features = np.array(train_features, np.float32)
        self.mean = np.mean(train_features, axis=0).reshape((1, -1))
        self.std = np.std(train_features, axis=0).reshape((1, -1)) + 10 ** -6
        train_features -= self.mean
        train_features /= self.std

        np.save("%s/mean.npz" % self.data_dir, self.mean)
        np.save("%s/std.npz" % self.data_dir, self.std)

        # Split of some validation data
        if not hasattr(self, 'indices'):  # use existing splits if there are
            np.random.seed(0)
            self.indices = np.random.permutation(np.arange(len(train_features)))
        split_idx = int(len(train_features) * train_ratio)
        val_features = train_features[self.indices[split_idx:]]
        val_labels = train_labels[self.indices[split_idx:]]
        train_features = train_features[self.indices[:split_idx]]
        train_labels = train_labels[self.indices[:split_idx]]

        # Start training
        for epoch in range(0, epochs):
            start = time.time()
            log('>>>> Starting epoch %d (it: %d)' % (epoch, self.current_step))
            sys.stdout.flush()

            # Randomize ordeer
            order = np.random.permutation(np.arange(len(train_features)))
            train_features = train_features[order]
            train_labels = train_labels[order]

            # Start threads to fill sample queue
            queue = self._generator_queue(train_features, train_labels, batch_size)
            while True:
                data = queue.get(timeout=60)
                if data is not self.sentinel:
                    # Do a training step to learn to corrently score the solution (predicted labels)
                    features, true_lbls = data
                    self.current_step, _ = self.sess.run(
                        [self.global_step, self.train_step],
                        feed_dict={self.features_pl: features,
                                   self.true_lbls: true_lbls})

                    if self.current_step % 100 == 0:
                        # This is just for reporting to tensorboard
                        f1 = self.get_f1(features, true_lbls)
                        summary_str, mean_loss, mean_train_f1 = self.sess.run([self.summary_op,
                                                                               tf.reduce_mean(self.loss),
                                                                               tf.reduce_mean(self.gt_scores_pl[:, 1])],
                                                                              feed_dict={self.features_pl: features,
                                                                                         self.gt_scores_pl: f1,
                                                                                         self.true_lbls: true_lbls})
                        self.writer.add_summary(summary_str, self.current_step)
                        self.writer.flush()

                        log('Step %d: Loss: %.3f, Train-F1: %.3f' % (
                            self.current_step, mean_loss, mean_train_f1))
                else:
                    break

            # store model at the end of each epoch
            if self.saver:
                self.saver.save(self.sess, '%s/weights' % self.data_dir, global_step=self.current_step)
            log(">>>> Epoch took %.2f seconds" % (time.time() - start))
            sys.stdout.flush()

            f1_scores = []

            for idx in range(0, len(val_features), batch_size):
                # Get a batch
                features = val_features[idx:min(len(val_features), idx + batch_size)]
                gt_labels = val_labels[idx:min(len(val_labels), idx + batch_size)]

                f1 = self.get_f1(features, gt_labels)
                f1_scores.extend(list(f1))

            if len(f1_scores) > 0:
                f1_scores = np.array(f1_scores)
                log('>>>> Validation mean F1: %.3f' % np.mean(f1_scores[:, 1]))
                sys.stdout.flush()
                summary_str = self.sess.run(self.gt_dist_op, feed_dict={self.gt_scores_pl: f1_scores})
                if self.val_writer:
                    self.val_writer.add_summary(summary_str, self.current_step)
                    self.val_writer.flush()

        if len(f1_scores) > 0:
            return np.mean(f1_scores[:, 1])
        else:
            return None

    def get_z(self, feature_input, reuse=False):

        with tf.variable_scope("pred_scope_z", reuse=reuse):
            W = tf.get_variable("W", shape=[self.feature_dim, self.num_hidden],
                                initializer=tf.glorot_normal_initializer())
            b = tf.get_variable("b", initializer=tf.zeros([self.num_hidden]))
            self.loss += self.weight_decay * self.regularizer(W)

            p1 = self.nonlinearity(tf.matmul(feature_input, W) + b)

            W2 = tf.get_variable("W2", shape=[self.num_hidden, self.num_indicators],
                                 initializer=tf.glorot_normal_initializer())
            b2 = tf.get_variable("b2", initializer=tf.zeros([self.num_indicators]))

            p2 = tf.matmul(p1, W2) + b2
            self.loss += self.weight_decay * self.regularizer(W2)

            return tf.nn.softmax(p2)

    def get_f1(self, features, gt_labels):
        yhats = self.sess.run(self.y_hats, feed_dict={self.features_pl: features,
                                                      self.true_lbls: gt_labels})

        f1s = np.zeros((gt_labels.shape[0], 2))
        f1s[:, 1] = [self.gt_value(yhats[-1][idx], gt_labels[idx]) for idx in
                     np.arange(0, gt_labels.shape[0])]
        f1s[:, 0] = 1 - f1s[:, 1]

        return f1s

    def gt_value(self, pred_labels, gt_labels):

        pred_labels = np.array(pred_labels >= 0.5, np.float32)

        intersect = np.sum(np.min([pred_labels, gt_labels], axis=0))
        union = np.sum(np.max([pred_labels, gt_labels], axis=0))
        return 2 * intersect / float(intersect + max(10 ** -8, union))

    def get_prediction(self, feature_input, label_input, reuse=True):

        with tf.variable_scope("pred_scope", reuse=reuse):
            W = tf.get_variable("W", shape=[self.feature_dim, self.num_hidden],
                                initializer=tf.glorot_normal_initializer())
            b = tf.get_variable("b", initializer=tf.zeros([self.num_hidden]))
            W2 = tf.get_variable("W2", shape=[self.num_hidden, self.label_dim],
                                 initializer=tf.glorot_normal_initializer())
            b2 = tf.get_variable("b2", initializer=tf.zeros([self.label_dim]))

            self.loss += self.weight_decay * self.regularizer(W)
            self.loss += self.weight_decay * self.regularizer(W2)

            p1 = tf.matmul(feature_input, W) + b
            self.prediction1 = tf.matmul(self.nonlinearity(p1), W2) + b2

            prediction2 = tf.reduce_sum(label_input * self.prediction1, axis=1)

            # We add global features as done in "Structured Prediction Energy Networks", Eq (5)
            Wp = tf.get_variable("Wp", shape=[self.label_dim, self.num_pairwise],
                                 initializer=tf.glorot_normal_initializer())
            Wp2 = tf.get_variable("Wp2", shape=[self.num_pairwise, 1],
                                  initializer=tf.glorot_normal_initializer())
            prior_prediction = tf.matmul(label_input, Wp)
            prior_prediction = self.nonlinearity(prior_prediction)
            prior_prediction = tf.squeeze(tf.matmul(prior_prediction, Wp2))
            prediction2 += prior_prediction

            if not reuse:
                self.loss += self.weight_decay * self.regularizer(Wp)
                self.loss += self.weight_decay * self.regularizer(Wp2)
                variable_summaries(Wp, 'Wp')
                variable_summaries(Wp2, 'Wp2')
                variable_summaries(W, 'W')
                variable_summaries(b, 'b')

        return prediction2

    def build_graph(self):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.features_pl = tf.placeholder(tf.float32, [None, self.feature_dim])
        self.gt_scores_pl = tf.placeholder(tf.float32, [None, 2])
        self.true_lbls = tf.placeholder(tf.float32, [None, self.label_dim])
        self.loss = 0

        self.prediction1 = None
        _ = self.get_prediction(self.features_pl, tf.zeros(tf.shape(self.true_lbls)), reuse=False)

        self.y_hats = self.backprop_proj_y()

        self.loss += self.loss_function(self.y_hats, self.true_lbls)
        self.train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(tf.reduce_mean(self.loss),
                                                                                 global_step=self.global_step)
        # Create summary operations
        tf.summary.scalar('loss', tf.reduce_mean(self.loss))
        self.gt_dist_op = tf.summary.scalar('gt_f1_scores', tf.reduce_mean(self.gt_scores_pl[:, 1]))
        self.summary_op = tf.summary.merge_all()

    def _generator_queue(self, train_features, train_labels, batch_size, num_threads=5):

        queue = Queue.Queue(maxsize=20)

        # Build indices queue to ensure unique use of each batch
        indices_queue = Queue.Queue()
        for idx in np.arange(0, len(train_features), batch_size):
            indices_queue.put(idx)

        def generate():
            try:
                while True:
                    idx = indices_queue.get_nowait()
                    features = train_features[idx:min(len(train_features), idx + batch_size)]
                    gt_labels = train_labels[idx:min(len(train_labels), idx + batch_size)]

                    queue.put((features, gt_labels))
            except Queue.Empty:
                queue.put(self.sentinel)

        for _ in range(num_threads):
            thread = threading.Thread(target=generate)
            thread.start()

        return queue

    def predict(self, features):
        features = np.array(features, np.float64)
        features -= self.mean[0]
        features /= self.std[0]

        labels = self.sess.run(self.y_hats,
                               feed_dict={self.features_pl: [features]})[-1].flatten()

        return labels >= 0.5

    def backprop_proj_y(self):
        yhats, scores = [], []
        indices = tf.tile([[i + 1. for i in range(self.num_indicators)]],
                          tf.stack([tf.shape(self.prediction1)[0], 1]))
        Z = tf.reduce_sum(self.get_z(self.features_pl) * indices, -1)
        u_curr, v = tf.nn.sigmoid(self.prediction1), 0

        for i in range(self.inner_iter):
            score = self.get_prediction(self.features_pl, u_curr)
            grads = tf.gradients(score, u_curr)[0]
            v = self.inf_momentum * v + self.inf_lr * grads
            u_curr += v
            b, p, q = u_curr, 0, 0

            # Dykstra's alternating projection
            for k in range(self.inf_r):
                # project onto set C
                a = tf.minimum(b + p, 1)
                p += b - a

                # project onto set D
                b = self.proj_simplex(a + q, Z)
                q += a - b

            u_curr = b
            yhats.append(u_curr)

        return yhats

    # shape(y) = [batch, L] = shape(output)
    def proj_simplex(self, y, Z):

        # obtain tensor of the sorted label values
        # Note - top_k is differentiable with regards to sorted values, not indices.
        sorted_y, _ = tf.nn.top_k(y, k=self.label_dim)

        # get the array of cumulative sums of a sorted (decreasing) copy of v
        cumsum_y = tf.cumsum(sorted_y, -1)

        # make 1..L indices for each y entry, to be used in cond
        indices = tf.tile([tf.range(1, self.label_dim + 1, dtype=tf.float32)],
                          tf.stack([tf.shape(y)[0], 1]))

        Z_ = tf.tile(tf.expand_dims(Z, 1), [1, self.label_dim])
        cond = tf.nn.softsign(tf.nn.relu(sorted_y * indices - (cumsum_y - Z_)))

        # get the number of > 0 components of the optimal solution
        rho = tf.nn.softmax(cond * indices, -1)

        theta = (tf.reduce_sum(cumsum_y * rho, -1) - Z) * (1 / tf.reduce_sum(rho * indices, -1))
        theta = tf.tile(tf.expand_dims(theta, -1), [1, self.label_dim])

        return tf.nn.relu(y - theta)

    def f1(self, y, true_y):
        tp = tf.reduce_sum(true_y * y, axis=-1)
        return 2. * tp / (tf.reduce_sum(y + true_y, axis=-1))

    # yhats = [T, batch, L]
    def loss_function(self, yhats, label):
        final_loss = 0
        for i, prob in enumerate(yhats):
            loss_sum = -self.f1(prob, label)

            final_loss += ((1. / (self.inner_iter - i + 1.)) * loss_sum)

        loss = tf.reduce_sum(final_loss, 0) / self.inner_iter
        return loss


def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))


def log(msg):
    print("{} {}".format(time.ctime(), msg))
