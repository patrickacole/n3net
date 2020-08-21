#!/bin/bash

# Not sure what this does?
nvidia-smi

# Activate desired environment
# Need to source bashrc for some reason
source ~/.bashrc
conda activate pytorch-env

# Check to see if data is in the right spot
if [[ ! -d "/data/pacole2" ]]
then
    mkdir /data/pacole2
fi

if [[ ! -d "/data/pacole2/DeepLesionTestPreprocessed/" ]]
then
    echo "Data is not on gpu storage"
    echo "Copying over data from shared storage"
    cp /shared/rsaas/pacole2/DeepLesionTestPreprocessed.zip /data/pacole2/

    cd /data/pacole2/
    unzip -qq DeepLesionTestPreprocessed.zip
    rm DeepLesionTestPreprocessed.zip
    cd /home/pacole2/
fi

# Data is ready now run python file
cd ~/Projects/n3net/src_denoising/
echo "Running python script now"
python -u main.py --eval --eval_epoch 51 --evaldir /home/pacole2/Projects/n3net/results_deeplesion_denoising/0009-/ # add other arguments
