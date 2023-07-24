import tensorflow as tf
import numpy as np
import sys


class MLPHierarchicalModel():
    """Class for hierarchical mlp models"""

    def __init__(self, input_dim, num_neuron, num_block, num_layer_pb, lambda_value, use_linear, decay, verbose,
                 random_seed=1):
        """
        Args:
                - input_dim: [int] number of configurations for the dataset (i.e., column dimension)
                - num_neuron: [int] number of neurons in each MLP layer
                - num_block: [int] number of blocks in the network
                - num_layer_pb: [int] number of layers in per block
                - lambda_value: [float] the lambda value
                - use_linear: [bool] whether to use linear models
                - decay: [float] fraction to decay learning rate
                - verbose: whether print the intermediate results
                - random_seed: [int] the random seed to consider
        """
        self.input_dim = input_dim
        self.num_neuron = num_neuron
        self.num_block = num_block
        self.num_layer_pb = num_layer_pb
        self.lambda_value = lambda_value
        self.use_linear = use_linear
        self.decay = decay
        self.verbose = verbose
        self.name = 'MLPHierarchicalModel'
        self.random_seed = random_seed

        tf.reset_default_graph()  # Saveguard if previous model was defined
        tf.set_random_seed(random_seed)  # Set tensorflow seed for paper replication

    def __build_neural_net(self):
        input_layer = self.X
        output = None
        for block_id in range(self.num_block):
            backcast, forecast = self.__create_block(input_layer)
            input_layer = tf.concat([input_layer, backcast], 1)
            if block_id == 0:
                output = forecast
            else:
                output = output + forecast

        if self.use_linear:
            linear_input = self.X
            linear_output = tf.layers.dense(linear_input, 1, kernel_regularizer=tf.contrib.layers.l2_regularizer(
                float(self.lambda_value)))
            output = output + linear_output

        return output

    def __create_block(self, x):
        layer = x
        for i in range(self.num_layer_pb):
            if i == 0:
                layer = tf.layers.dense(layer, self.num_neuron, tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.random_seed),
                                        kernel_regularizer=tf.contrib.layers.l1_regularizer(float(self.lambda_value)))
            else:
                layer = tf.layers.dense(layer, self.num_neuron, tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.random_seed))
        backcast = tf.layers.dense(layer, self.input_dim, tf.nn.relu)
        forecast = tf.layers.dense(layer, 1)

        return backcast, forecast

    def build_train(self):
        """Builds model for training"""
        self.__add_placeholders_op()
        self.__add_pred_op()
        self.__add_loss_op()
        self.__add_train_op()

        self.init_session()

    def __add_placeholders_op(self):
        """ Add placeholder attributes """
        self.X = tf.placeholder("float", [None, self.input_dim])
        self.Y = tf.placeholder("float", [None, 1])
        self.lr = tf.placeholder("float")  # to schedule learning rate

    def __add_pred_op(self):
        """Defines self.pred"""
        self.output = self.__build_neural_net()

    def __add_loss_op(self):
        """Defines self.loss"""
        l2_loss = tf.losses.get_regularization_loss()
        self.loss = l2_loss + tf.losses.mean_squared_error(self.Y, self.output)

    def __add_train_op(self):
        """Defines self.train_op that performs an update on a batch"""
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grads, vs = zip(*optimizer.compute_gradients(self.loss))
            grads, gnorm = tf.clip_by_global_norm(grads, 1)
            self.train_op = optimizer.apply_gradients(zip(grads, vs))

    def init_session(self):
        """Defines self.sess, self.saver and initialize the variables"""
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.8))))
        # self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def finalize(self):
        self.sess.close()
        tf.get_default_graph().finalize()
