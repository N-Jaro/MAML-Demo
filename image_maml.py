import numpy as np
import tensorflow as tf

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops


class BaseModel:
    def __init__(self, img_size, dim_input, dim_output, filter_num, update_lr,
                 meta_lr, meta_batch_size, update_batch_size, test_num_updates):
        """ must call construct_model() after initializing MAML! """
        self.img_size = img_size    # tuple
        self.dim_input = dim_input
        self.channels = dim_output
        self.dim_output = dim_output
        self.filter_num = filter_num

        self.update_lr = update_lr
        self.meta_lr = meta_lr
        self.update_batch_size = update_batch_size
        self.test_num_updates = test_num_updates
        self.meta_batch_size = meta_batch_size

        self.inputa = tf.placeholder(tf.float32)
        self.inputb = tf.placeholder(tf.float32)
        self.labela = tf.placeholder(tf.float32)
        self.labelb = tf.placeholder(tf.float32)

    def update(self, loss, weights):
        grads = tf.gradients(loss, list(weights.values()))
        gradients = dict(zip(weights.keys(), grads))
        new_weights = dict(
            zip(weights.keys(), [weights[key] - self.update_lr * gradients[key] for key in weights.keys()]))
        return new_weights

    def construct_cnn(self):
        weights = {}
        dtype = tf.float32
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        k = 3

        weights['conv1'] = tf.get_variable('conv1', [k, k, self.channels, self.filter_num],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b_conv1'] = tf.Variable(tf.zeros([self.filter_num]))

        weights['conv2'] = tf.get_variable('conv2', [k, k, self.filter_num, self.filter_num],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b_conv2'] = tf.Variable(tf.zeros([self.filter_num]))

        weights['conv3'] = tf.get_variable('conv3', [k, k, self.filter_num, self.filter_num],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b_conv3'] = tf.Variable(tf.zeros([self.filter_num]))

        weights['fc1'] = tf.Variable(tf.random_normal([self.filter_num, self.dim_output]), name='fc1')

        weights['b_fc1'] = tf.Variable(tf.zeros([self.dim_output]))

        return weights

    def cnn(self, inp, weights):
        def conv_block(cinp, cweight, bweight, activation):
            """ Perform, conv, batch norm, nonlinearity, and max pool """
            stride, no_stride = [1, 2, 2, 1], [1, 1, 1, 1]
            conv_output = tf.nn.conv2d(cinp, cweight, no_stride, 'SAME') + bweight
            return activation(conv_output)

        inp = tf.reshape(inp, [-1, self.img_size[0], self.img_size[1], self.channels])
        hidden1 = conv_block(inp, weights['conv1'], weights['b_conv1'], tf.nn.relu)
        hidden2 = conv_block(hidden1, weights['conv2'], weights['b_conv2'], tf.nn.relu)
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b_conv3'], tf.nn.relu)
        return hidden3

    def forward_cnn(self, inp, weights):
        inp = tf.reshape(inp, [-1, self.dim_input])

        cnn_outputs = self.cnn(inp, weights)
        return cnn_outputs


class MetaCNN(BaseModel):
    def __init__(self, img_size, dim_input, dim_output, filter_num, update_lr,
                 meta_lr, meta_batch_size, update_batch_size,
                 test_num_updates):
        print("Initializing MetaCNN...")
        BaseModel.__init__(self, img_size, dim_input, dim_output, filter_num, update_lr,
                 meta_lr, meta_batch_size, update_batch_size, test_num_updates)

    def loss_func(self, pred, label):
        pred = tf.reshape(pred, [-1])
        label = tf.reshape(label, [-1])
        return tf.reduce_mean(tf.square(pred - label))

    def construct_model(self):
        with tf.variable_scope('model', reuse=None):
            with tf.variable_scope('maml', reuse=None):
                self.weights = weights = self.construct_cnn()

            num_updates = self.test_num_updates

            def task_metalearn(inp):
                """ Perform gradient descent for one task in the meta-batch. """
                inputa, inputb, labela, labelb = inp
                task_outputbs, task_lossesb = [], []

                task_outputa = self.forward(inputa, weights)  # only reuse on the first iter
                task_lossa = self.loss_func(task_outputa, labela)

                fast_weights = self.update(task_lossa, weights)

                output = self.forward(inputb, fast_weights)
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, labelb))

                for j in range(num_updates - 1):
                    loss = self.loss_func(self.forward(inputa, fast_weights), labela)
                    fast_weights = self.update(loss, fast_weights)

                    output = self.forward(inputb, fast_weights)
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, labelb))

                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]
                return task_output

            out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates]

            inputs = (self.inputa, self.inputb, self.labela, self.labelb)
            result = tf.map_fn(task_metalearn,
                               elems=inputs,
                               dtype=out_dtype,
                               parallel_iterations=self.meta_batch_size)
            outputas, outputbs, lossesa, lossesb = result

        # Performance & Optimization
        self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(self.meta_batch_size)
        self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(self.meta_batch_size)
                                              for j in range(num_updates)]
        self.total_rmse1 = tf.sqrt(lossesa)
        self.total_rmse2 = [tf.sqrt(total_losses2[j]) for j in range(num_updates)]

        self.outputas, self.outputbs = outputas, outputbs
        self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)
        self.metatrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_losses2[num_updates-1])

        maml_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "model/maml")
        self.finetune_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1, var_list=maml_vars)

    def forward(self, inp, weights):
        cnn_outputs = self.forward_cnn(inp, weights)
        preds = tf.nn.sigmoid(tf.matmul(cnn_outputs, weights['fc1']) + weights['b_fc1'])
        return preds

