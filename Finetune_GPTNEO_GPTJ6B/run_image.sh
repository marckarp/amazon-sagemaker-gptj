#!/bin/bash
dir_to_mount=$(pwd)
docker run -it --ipc=host --gpus all -v /home/ec2-user/SageMaker/.cache:/root/.cache -v $dir_to_mount:/workspace gpt
