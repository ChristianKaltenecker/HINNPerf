from typing import Any

from models.models import MLPHierarchicalModel
import numpy as np
import time
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error


class ModelRunner:
    """
    Class for learning HINNPerf models.
    This class exposes some methods that are needed by scikit-learn in order to be able to use methods from scikit-learn.
    For instance, we use 'cross_validate' to perform a 5-fold cross-validation.
    """

    def __init__(self, num_neuron=128, num_block=4, num_layer_pb=3, lambda_value=0.1, decay=None, use_linear=False, lr=0.001, verbose=False, random_state=1, model_class=MLPHierarchicalModel):
        """
        Initializes the class using the given configuration. Note that the values are the supposedly default values used
        in the original HINNPerf paper.

        Args:
            num_neuron: The number of neurons to use.
            num_block: The number of blocks to use for learning.
            num_layer_pb: The number of layers to use for learning.
            lambda_value: The lambda value.
            decay: The threshold for the decay.
            use_linear: Whether linear models should be learned too.
            lr: The minimum learning rate threshold.
            verbose: Whether more output should be generated.
            random_state: The random seed used for learning.
            model_class: The model class to use.
        """
        self.num_neuron = num_neuron
        self.num_block = num_block
        self.num_layer_pb = num_layer_pb
        self.lambda_value = lambda_value
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
            x_train: The set of configurations to train on
            y_train: The set of performance values of the configurations to train on
        """
        self.input_dim = x_train.shape[1]
        self.model = self.execute_learning(x_train, y_train)

        #self.model.finalize()

        return self

    def train(self, x_train, y_train):
        """
        Train the model

        Args:
            x_train: The set of configurations to train on
            y_train: The set of performance values of the configurations to train on
        """

        self.input_dim = x_train.shape[1]
        self.model = self.execute_learning(x_train, y_train)

        Y_pred_train = self.model.sess.run(self.model.output, {self.model.X: x_train})

        mape_error = mean_absolute_percentage_error(y_train, Y_pred_train)

        return mape_error

    def execute_learning(self, x_train, y_train):
        """"
        Executes the learning procedure using the given data.

        Args:
            x_train: The set of configurations to train on
            y_train: The set of performance values of the configurations to train on
        """
        self.input_dim = x_train.shape[1]
        self.model = self.model_class(self.input_dim, self.num_neuron, self.num_block, self.num_layer_pb, self.lambda_value,
                                      self.use_linear, self.decay, self.verbose, self.random_state)
        self.model.build_train()
        lr = self.lr
        decay = lr / 1000
        for epoch in range(1, 2000):
            _, cur_loss, pred = self.model.sess.run([self.model.train_op, self.model.loss, self.model.output],
                                               {self.model.X: x_train, self.model.Y: y_train, self.model.lr: lr})

            lr = lr * 1 / (1 + decay * epoch)
        return self.model

    def predict(self, x) -> Any:
        """
        Predicts the given configurations and returns a vector of performance values.

        Args:
            x: The data to predict on.
        """
        return self.model.sess.run(self.model.output, {self.model.X: x})

    def finalize(self):
        """
        This method finalizes the learning procedure. That is, the tensorflow session is ended.
        Note that the model will be removed as soon as the session is ended and, thus, cannot be used further.
        """
        self.model.finalize()

    def get_params(self, deep=False):
        """
        This method returns the parameters of the model. This method is needed for cross_validate from scikit-learn.
        Returns:
            a dictionary containing all the options that can be changed for hyperparameter tuning.
        """
        return {"num_block": self.num_block, "num_layer_pb": self.num_layer_pb, "num_neuron": self.num_neuron, "lamda": self.lambda_value, "random_state": self.random_state}

    def set_params(self, **parameters):
        """
        This method sets the given parameters of the model. Note that we assume that the parameters exist.
        If not, an error will be raised.
        Returns:
            The current instance.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
