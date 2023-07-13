from typing import Any

from models.models import MLPHierarchicalModel
from utils.data_preproc import system_samplesize, seed_generator, DataPreproc
import numpy as np
import time
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error


class ModelRunner():
    """Generic class for training models"""

    def __init__(self, model_class, config):
        """
        Args:
            data_preproc: [DataPreproc object] preprocess and generate training data
            model_class: [class] deep learning model class
            config: configures to define model, which should contain:
                        - input_dim: [int] number of configurations for the dataset (i.e., column dimension)
                        - num_neuron: [int] number of neurons in each MLP layer
                        - num_block: [int] number of blocks in the network
                        - num_layer_pb: [int] number of layers in per block
                        - decay: [float] fraction to decay learning rate
                        - verbose: whether print the intermediate results
                        - random_state: The random seed for the learner
        """
        self.data_preproc = None

        self.input_dim = config['input_dim']
        self.num_neuron = config['num_neuron']
        self.num_block = config['num_block']
        self.num_layer_pb = config['num_layer_pb']
        self.lamda = config['lamda']
        self.use_linear = config['linear']
        self.decay = config['decay']
        self.lr = config['lr']
        self.verbose = config['verbose']
        if 'random_state' in config:
            self.random_seed = config['random_state']
        else:
            self.random_seed = 1

        self.model_class = model_class
        self.model = None

    def fit(self, x_train, y_train):
        """
        Fit the model

        Args:
            config: configures to create a model object
        """

        model = self.execute_learning(x_train, y_train)

        model.finalize()

        return self

    def train(self, x_train, y_train):
        """
        Train the model

        Args:
            config: configures to create a model object
        """

        self.model = self.execute_learning(x_train, y_train)

        Y_pred_train = self.model.sess.run(self.model.output, {self.model.X: x_train})

        mape_error = mean_absolute_percentage_error(y_train, Y_pred_train)

        return mape_error

    def execute_learning(self, x_train, y_train):
        model = self.model_class(self.input_dim, self.num_neuron, self.num_block, self.num_layer_pb, self.lamda,
                                 self.use_linear, self.decay, self.verbose, self.random_seed)
        model.build_train()
        lr = self.lr
        decay = lr / 1000
        train_seed = 0
        for epoch in range(1, 2000):
            train_seed += 1
            _, cur_loss, pred = model.sess.run([model.train_op, model.loss, model.output],
                                               {model.X: x_train, model.Y: y_train, model.lr: lr})

            lr = lr * 1 / (1 + decay * epoch)
        return model

    def predict(self, x) -> Any:
        return self.model.sess.run(self.model.output, {self.model.X: x})

    def finalize(self):
        self.model.finalize()
