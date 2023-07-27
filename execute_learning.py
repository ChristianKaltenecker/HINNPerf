#!/bin/env python3

import os
import sys
from pathlib import Path
from time import perf_counter
from typing import List, Dict, Any

import logging
import numpy as np
import pandas as pd

from models.models import MLPHierarchicalModel
from models.model_runner import ModelRunner
from sklearn.model_selection import ParameterGrid, cross_validate
from sklearn.metrics import mean_absolute_percentage_error


class AutomationScriptParser:
    """
    This class handles parsing the automation script (see the SPLConqueror project).
    Note that we only support a very limited number of commands.
    """

    """
    The parameters for the approach.
    """
    PARAMETERS = dict(
        num_neuron=[128, int],
        num_block=[4, int],
        num_layer_pb=[3, int],
        lambda_value=[0.1, float],
        use_linear=[False, bool],
        lr=[0.001, float],
        decay=[None, float],
        verbose=[True, bool],
        random_state=[1, int],
    )

    def __init__(self):
        """
        Initializes the script parser.
        """
        self.y_evaluation = None
        self.x_evaluation = None
        self.config_num = None
        self.y_train = None
        self.x_train = None
        self.param_space = None
        self.nfp = None
        self.config = dict(
            num_neuron=128,
            num_block=4,
            num_layer_pb=3,
            lambda_value=0.1,
            use_linear=False,
            lr=0.001,
            decay=None,
            verbose=True,
            random_state=1
        )
        self.grid_search_file_name = None

    def read_in_settings(self, settings: List[str], parameter_list: Dict[str, List]) -> Dict[str, Any]:
        """
        Reads in the given settings for hyperparameter tuning.

        Args:
            settings: The settings provided by the user.
            parameter_list: The current parameter list that will be overwritten and used to check whether the user
                has inserted the correct datatypes.
        Returns:
            A dictionary containing the parameters and their values.
        """
        for setting in settings:
            setting_value_pair = setting.split(":")
            if setting_value_pair[0] in parameter_list.keys():
                key = setting_value_pair[0]
                try:
                    if len(parameter_list[key]) > 1:
                        # Convert it to the datatype given in the tuple
                        if parameter_list[key][1] is int:
                            # In the case a float is inserted as a string, the string can not be directly converted to int.
                            # First, convert it to float and afterwards, to int
                            parameter_list[key][0] = parameter_list[key][1](float(setting_value_pair[1].strip()))
                        else:
                            parameter_list[key][0] = parameter_list[key][1](setting_value_pair[1].lower())
                    else:
                        parameter_list[key][0] = setting_value_pair[1]
                except Exception as err:
                    print(
                        f"Parameter {setting_value_pair[0]} is not of the right type.\n {err}",
                        file=sys.stderr, flush=True)
            else:
                print(f"Parameter {setting_value_pair[0]} is unknown.\n", file=sys.stderr, flush=True)
        parameter_to_value_dict = {}
        for dict_key in parameter_list.keys():
            parameter_to_value_dict[dict_key] = parameter_list[dict_key][0]
        return parameter_to_value_dict

    def parse_automation_file(self, path: str) -> None:
        """
        This method parses the automation file from the given path.

        Args:
            path: The path to the automation file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file {path} does not exist!")
        # Change working directory to location of the automation script
        os.chdir(os.path.dirname(path))
        with open(path, 'r') as automation_file:
            for line in automation_file:
                arguments = line.replace("\n", "").split(" ")[1:]
                # Ignore comments
                if line.startswith("#") or line.strip() == "":
                    continue
                elif line.startswith("all "):
                    if len(arguments) != 1:
                        raise SyntaxError("The argument list of 'all' should contain only one element" +
                                          " -- the path to the csv file.")
                    self.parse_measurement_file(arguments[0])
                elif line.startswith("evaluationset "):
                    if len(arguments) != 1:
                        raise SyntaxError("The argument list of 'evaluationset' should contain only one element" +
                                          " -- the path to the csv file.")
                    self.parse_evaluation_set_file(arguments[0])
                elif line.startswith("learn-opt"):
                    self.execute_hyperparameter_learning(arguments)
                elif line.startswith("learn"):
                    self.config = self.read_in_settings(arguments, self.PARAMETERS)
                    self.learn_model()
                else:
                    raise SyntaxError(f"The line {line} is not supported.")

    def parse_measurement_file(self, path: str) -> None:
        """
        Parses the measurement file at the given path.

        Args:
            path: The path to the measurement file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file {path} does not exist!")
        # Save the NFP column; this column is needed later on for the evaluation set
        with open(path, 'r') as data_file:
            self.nfp = data_file.readline().replace("\n", "").strip().split(";")[-1]
        whole_data = np.genfromtxt(path, delimiter=';', skip_header=1)
        (self.all_sample_num, config_num) = whole_data.shape
        self.config_num = config_num - 1

        self.x_train = whole_data[:, 0:self.config_num]
        self.y_train = whole_data[:, self.config_num][:, np.newaxis]

    def parse_evaluation_set_file(self, path: str) -> None:
        """
        Parses the evaluation file at the given path.

        Args:
            path: The path to the evaluation file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file {path} does not exist!")
        # Detect the correct NFP column
        with open(path, 'r') as evaluation_file:
            header_elements = evaluation_file.readline().replace("\n", "").strip().split(';')
            nfp_position = header_elements.index(self.nfp)
        self.evaluation_set = np.genfromtxt(path, delimiter=';', skip_header=1)
        self.x_evaluation = self.evaluation_set[:, 0:self.config_num]
        self.y_evaluation = self.evaluation_set[:, nfp_position][:, np.newaxis]

    def learn_model(self) -> None:
        """
        This method performs the typical learning procedure without hyperparameter tuning.
        This method should be used only after the best hyperparameter setting has been discovered.
        """
        # Initialize the learner
        runner = ModelRunner(**self.config)
        start = perf_counter()
        runner.train(self.x_train, self.y_train)
        elapsed = perf_counter() - start
        error = mean_absolute_percentage_error(self.y_evaluation, runner.predict(self.x_evaluation))
        print(f"Elapsed learning time(seconds): {elapsed}")
        print(f"Prediction Error: {error * 100} %")

    def execute_hyperparameter_learning(self, arguments: List[str]) -> None:
        """
        This method performs a grid search on the given parameter space.
        Note that GridSearchCV from sklearn does the same, but uses (1) another scoring function and (2) does not allow
        to access the individual models to perform predictions on an optional evaluation set.

        Args:
            arguments: The arguments provided by the user for hyperparameter tuning.
        """
        self.change_parameter_space(arguments)
        all_parameter_settings = ParameterGrid(self.param_space)
        parameters: Dict[str, List] = dict()
        parameters["params"] = []
        parameters["error"] = []
        for setting in all_parameter_settings:
            # Apply setting
            for parameter, value in setting.items():
                # First, check if the parameter exists
                if parameter not in self.config:
                    raise SystemError(f"Parameter {parameter} does not exist.")
                self.config[parameter] = value

            # Perform k-fold cross validation
            results = cross_validate(ModelRunner(**self.config), self.x_train, self.y_train, cv=5,
                                     return_estimator=True, scoring='neg_mean_absolute_percentage_error')
            error = 0
            for estimator in results['estimator']:
                if len(self.x_evaluation) > 0:
                    error += mean_absolute_percentage_error(self.y_evaluation, estimator.predict(self.x_evaluation))
                    estimator.finalize()
                else:
                    raise SystemError("Please define an evaluation set for hyperparameter tuning.")
            parameters["params"].append(setting)
            error = error / 5

            # This error value is in percent
            parameters["error"].append(error * 100)

        # Create pandas dataframe and extract it
        if self.grid_search_file_name != "":
            df = pd.DataFrame(columns=list(parameters['params'][0].keys()) + ['error'])
            for i in range(0, len(parameters['params'])):
                df.loc[len(df)] = list(parameters['params'][i].values()) + [
                    np.abs(parameters['error'][i])]
            columns = list(df.columns[:-1])  # all columns except for error
            if 'random_state' in columns:
                columns.remove('random_state')
            df = df.groupby(columns).agg({'error': 'mean'})
            new_path = Path(Path.cwd(), self.grid_search_file_name).resolve()
            new_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(new_path, sep=';')

    def change_parameter_space(self, parameter_space_def: List[str]):
        """
        This method changes the parameter space as specified by the user.

        Args:
            parameter_space_def: The parameter space definition by the user.
        """
        if parameter_space_def is None or len(parameter_space_def) == 0:
            return

        if parameter_space_def[0].startswith("file:"):
            self.grid_search_file_name = parameter_space_def[0].split(":")[1]
            parameter_space_def = parameter_space_def[1:]
            if len(parameter_space_def) == 0:
                return
        to_execute = self.format_parameter_space(parameter_space_def)
        exec(to_execute, globals(), locals())

    def format_parameter_space(self, parameter_space: List[str]) -> str:
        """
        This method formats the given parameter space in order to use it as a command.

        Args:
            parameter_space: The parameter space specified by the user.
        Returns:
            A string that can be executed.
        """
        formatted_space = "self.param_space = {"
        for parameter in parameter_space:
            if ":" in parameter:
                name_value_pair = parameter.split(":")
                formatted_space += " '" + name_value_pair[0] + "' : "
                formatted_space += str(name_value_pair[1]).replace(';', ',') + ","
        formatted_space = formatted_space[:-1]
        formatted_space += "}"
        return formatted_space


def print_usage() -> None:
    """
    This method prints the usage of this script.
    """
    print("./execute_learning <PathToAFile>")
    print("PathToAFile\t represents the path to the automation file typically used for SPL Conqueror")
    print("Warning!\t Note that this script's only purpose is to read in a automation file and execute learning.")


def main() -> None:
    """
    The main method of this script.
    """
    if len(sys.argv) != 2:
        print_usage()
        exit(-1)
    automation_script_parser = AutomationScriptParser()
    path_to_automation_script = sys.argv[1]
    automation_script_parser.parse_automation_file(path_to_automation_script)


if __name__ == "__main__":
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    main()
