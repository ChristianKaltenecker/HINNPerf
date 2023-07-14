# HINNPerf

> :warning: **This is not the original HINNPerf tool. Please be adviced that the code of HINNPerf has changed in this repository to prepare HINNPerf for further comparison to other ML tools!**

You can find the original paper of HINNPerf along with a supplementary web site [here](https://dl.acm.org/doi/full/10.1145/3528100?casa_token=Z6hB1GSwELcAAAAA%3Alq27gjqh20zCF6tep1jox8tM1K1YvuaW97HmM0WmwLs5b3qVFL2XPxN0XMbn-wDM8WYHm1Xnrp0).

In principle, HINNPerf can only be executed in a Docker or podman container since HINNPerf uses a deprecated version of tensorflow.

## How to Execute HINNPerf

To execute HINNPerf inside a docker container, you can use a shell script that sets up a docker container using podman.
The syntax to execute the script is as follows:
'''
./execute_hinnperf_podman.sh -e <PathToEvaluationSetFile> -a <PathToAllMeasurementFile> -f <PathToFileForHyperparameterTuningResults> [--remove-images] <Hyperparameters>
'''
The meaning of the parameters is as follows:

| Name  | Description |
| :---: | :---------: |
| -e PathToEvaluationSetFile | The path to the evaluation set file. This file is used for testing. |
| -a PathToAllMeasurementFile | The path to the measurement set file to learn on. |
| -f PathToFileForHyperparameterTuningResults | This is an output file path and specified the path where the hyperparametertuning results should be written to. Currently, we support only exporting it to a csv file. If this parameter is provided, we perform hyperparameter tuning. |
| --remove-images | Whether the podman image should be removed or not. |

Note that you can use the flags in an arbitrary order.

For instance, you can use the following command:
'''
./execute_hinnperf_podman.sh -e /local/storage/kaltenec/tmp/brotli.csv -a /tmp/brotli/some_configs.csv -f /tmp/results.csv -l /tmp/hinnperf.log num_block:[2,3,4] random_state:[1,2,3]
'''

### Setup via Dockerfile (For Debugging)

Alternatively, you have the posibility to execute the docker container for debugging purposes.

To setup a container manually HINNPerf, we provide a [Dockerfile](./Dockerfile) for setting up a docker container.

To apply this file, we rely on docker and refer to the [documentation](https://docs.docker.com/install/linux/docker-ce/ubuntu/) on how to install docker on your Linux operating system.

After docker is installed, make sure that the docker daemon is running. On systemd, you can use ```systemctl status docker``` to check the status of the daemon and ```sudo systemctl start docker``` to start the daemon, if necessary.

Next, download the [Dockerfile](./Dockerfile).
The container is set up by invoking ```sudo docker build -t hinnperf ./``` in the directory where the Dockerfile is located.

After setting up the docker container, all required ressources (i.e., packages, programs, and scripts) are installed and can now be used inside the container.
To begin an interactive session, the command ```sudo docker run -i -t hinnperf /bin/bash``` can be used.

## Hyperparameter

| Name  | Description | Default Value |
| :---: | :---------: | :-----------: |
| num_block | Number of blocks (i.e., interactions) | [2,3,4,5] |
| num_layer_pb | Number of hidden layers in each block | [2,3,4] |
| lamda | The lambda for L1 regularization | [0.001,0.01,0.1,1,10] |
| random_state | Random seed that should be used | [1,2,3,4] |