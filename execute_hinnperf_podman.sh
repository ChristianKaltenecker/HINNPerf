#!/bin/bash

export XDG_RUNTIME_DIR="/scratch/$USER/"
export HOME="/scratch/$USER/"

# Definition of constants
PODMAN_ROOT="/local/storage/$USER/podman-images/"

CONTAINER_DATA_ROOT="/data"
CONTAINER_A_FILE_LOCATION="$CONTAINER_DATA_ROOT/learn.a"
CONTAINER_EVALUATION_FILE_LOCATION="$CONTAINER_DATA_ROOT/evaluation.csv"
CONTAINER_ALL_FILE_LOCATION="$CONTAINER_DATA_ROOT/all.csv"
CONTAINER_HYPERPARAMETER_FILE_LOCATION="$CONTAINER_DATA_ROOT/hyperparameter_results.csv"
CONTAINER_LOG_FILE="$CONTAINER_DATA_ROOT/result.log"
CONTAINER_LOG_ERROR_FILE="$CONTAINER_DATA_ROOT/result.log_error"

CONTAINER_SCRIPT_LOCATION="/application/HINNPerf/execute_learning.py"

A_FILE_LOCATION="/tmp/learn.a"

REMOVE_LOCAL_COPY=false

# Definition of functions
parse_arguments () {
    # Read in the arguments
    POSITIONAL_ARGS=()

    while [[ $# -gt 0 ]]; do
    case $1 in
        -e)
        EVALUATION_FILE="$2"
        shift # past argument
        shift # past value
        ;;
        -a)
        ALL_FILE="$2"
        shift # past argument
        shift # past value
        ;;
        -f)
        HYPERPARAMETER_CSV_RESULT_FILE="$2"
        shift # past argument
        shift # past value
        ;;
        -l)
        LOG_FILE="$2"
        shift # past argument
        shift # past value
        ;;
        --remove-images)
        REMOVE_LOCAL_COPY=true
        shift # past argument
        ;;
        -*|--*)
        echo "Unknown option $1"
        exit 1
        ;;
        *)
        POSITIONAL_ARGS+=("$1") # save positional arg
        shift # past argument
        ;;
    esac
    done

    LEARNING_OPTIONS="${POSITIONAL_ARGS[@]}"

    set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

    echo "Evaluation set file: $EVALUATION_FILE"
    echo "All file: $ALL_FILE"
    echo "Path to csv for hyperparameter tuning: $HYPERPARAMETER_CSV_RESULT_FILE"
    echo "Log file: $LOG_FILE"
    echo "Learning options: $LEARNING_OPTIONS"
    if [ ! -z $HYPERPARAMETER_CSV_RESULT_FILE ]
    then
        echo "Performing hyperparameter tuning!"
    else
        echo "Performing normal learning!"
    fi
    if [ "$REMOVE_LOCAL_COPY" = true ]
    then
        echo "Local copy will be removed afterwards"
    else
        echo "Local copy will NOT be removed afterwards"
    fi
}

create_a_file () {
    # Overwrite the a file
    echo "" > $A_FILE_LOCATION
    echo "all $CONTAINER_ALL_FILE_LOCATION" >> $A_FILE_LOCATION
    echo "evaluationset $CONTAINER_EVALUATION_FILE_LOCATION" >> $A_FILE_LOCATION
    if [ ! -z $HYPERPARAMETER_CSV_RESULT_FILE ]
    then
        echo "learn-opt file:$CONTAINER_HYPERPARAMETER_FILE_LOCATION $LEARNING_OPTIONS" >> $A_FILE_LOCATION
    else
        echo "learn $LEARNING_OPTIONS" >> $A_FILE_LOCATION
    fi
}

main () {
    # First, parse arguments
    parse_arguments $@

    # Create the automation script that is used by the HINNPerf implementation in the container
    create_a_file

    # After reading in, execute the script
    PODMAN_COMMAND="podman --root=$PODMAN_ROOT "

    if [ ! -d $PODMAN_ROOT ]
    then
        echo "Building the container locally"
        $PODMAN_COMMAND build -t hinnperf .
    fi

    # Start the container in detached mode
    echo "Starting the container"
    container_id=$($PODMAN_COMMAND run -d -t hinnperf /bin/bash)

    # Create the data root directory
    echo "Creating data root directory"
    $PODMAN_COMMAND exec -it $container_id mkdir -p $CONTAINER_DATA_ROOT

    # Copy the necessary files
    echo "Copying necessary data into the container"
    $PODMAN_COMMAND cp $A_FILE_LOCATION $container_id:$CONTAINER_A_FILE_LOCATION
    $PODMAN_COMMAND cp $ALL_FILE $container_id:$CONTAINER_ALL_FILE_LOCATION
    $PODMAN_COMMAND cp $EVALUATION_FILE $container_id:$CONTAINER_EVALUATION_FILE_LOCATION

    # Execution of the learning/hyperparameter tuning process
    # Redirect output to log file and log_error file
    echo "Executing learning procedure"
    echo "$PODMAN_COMMAND exec -it $container_id /bin/bash -c python $CONTAINER_SCRIPT_LOCATION  $CONTAINER_A_FILE_LOCATION > $CONTAINER_LOG_FILE 2> $CONTAINER_LOG_ERROR_FILE"
    $PODMAN_COMMAND exec -it $container_id /bin/bash -c "python $CONTAINER_SCRIPT_LOCATION  $CONTAINER_A_FILE_LOCATION > $CONTAINER_LOG_FILE 2> $CONTAINER_LOG_ERROR_FILE"

    # Copy log files
    echo "Copying results from container to destination"
    $PODMAN_COMMAND cp $container_id:$CONTAINER_LOG_FILE $LOG_FILE
    $PODMAN_COMMAND cp $container_id:$CONTAINER_LOG_ERROR_FILE ${LOG_FILE}_error
    $PODMAN_COMMAND cp $container_id:$CONTAINER_HYPERPARAMETER_FILE_LOCATION $HYPERPARAMETER_CSV_RESULT_FILE

    # Stop the container
    echo "Stopping container"
    $PODMAN_COMMAND container stop $container_id

    # Remove the container
    echo "Removing container"
    $PODMAN_COMMAND container rm $container_id

    if [ "$REMOVE_LOCAL_COPY" = true ]
    then
        echo "Removing local podman directory"
        $PODMAN_COMMAND rm -a
        $PODMAN_COMMAND image prune -a -f
        rm -r $PODMAN_ROOT
    fi
}

main $@