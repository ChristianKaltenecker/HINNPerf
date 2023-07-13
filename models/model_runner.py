from typing import Any

from models.models import MLPHierarchicalModel
import numpy as np
import time
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error


class ModelRunner():
    """Generic class for training models"""

    def __init__(self, num_neuron=128, num_block=4, num_layer_pb=3, lamda=0.1, decay=None, use_linear=False, lr=0.001, verbose=False, random_state=1, model_class=MLPHierarchicalModel):
        self.num_neuron = num_neuron
        self.num_block = num_block
        self.num_layer_pb = num_layer_pb
        self.lamda = lamda
        self.use_linear = use_linear
        self.lr = lr
        self.verbose = verbose
        self.random_state = random_state
        self.model_class = model_class
        self.decay = decay
        self.model = None

    def fit(self, x_train, y_train):
        """
        Fit the model

        Args:
            config: configures to create a model object
        """
        self.input_dim = x_train.shape[1]
        self.model = self.execute_learning(x_train, y_train)

        #self.model.finalize()

        return self

    def train(self, x_train, y_train):
        """
        Train the model

        Args:
            config: configures to create a model object
        """

        self.input_dim = x_train.shape[1]
        self.model = self.execute_learning(x_train, y_train)

        Y_pred_train = self.model.sess.run(self.model.output, {self.model.X: x_train})

        mape_error = mean_absolute_percentage_error(y_train, Y_pred_train)

        return mape_error

    def execute_learning(self, x_train, y_train):
        self.input_dim = x_train.shape[1]
        self.model = self.model_class(self.input_dim, self.num_neuron, self.num_block, self.num_layer_pb, self.lamda,
                                 self.use_linear, self.decay, self.verbose, self.random_state)
        self.model.build_train()
        lr = self.lr
        decay = lr / 1000
        train_seed = 0
        for epoch in range(1, 2000):
            train_seed += 1
            _, cur_loss, pred = self.model.sess.run([self.model.train_op, self.model.loss, self.model.output],
                                               {self.model.X: x_train, self.model.Y: y_train, self.model.lr: lr})

            lr = lr * 1 / (1 + decay * epoch)
        return self.model

    def predict(self, x) -> Any:
        return self.model.sess.run(self.model.output, {self.model.X: x})

    def finalize(self):
        self.model.finalize()

    def get_params(self, deep=False):
        return {"num_block": self.num_block, "num_layer_pb": self.num_layer_pb, "num_neuron": self.num_neuron, "lamda": self.lamda, "random_state": self.random_state}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
