#!/bin/bash

# Script to download model files using git-lfs

# Check if git is installed
if ! command -v git &> /dev/null
then
    echo "git could not be found, please install git before running this script."
    exit 1
fi

# Check if git-lfs is installed
if ! command -v git-lfs &> /dev/null
then
    echo "git-lfs could not be found, installing git-lfs..."
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get install -y git-lfs
fi

# Initialize git-lfs
git lfs install

# Clone the models repository
git clone https://huggingface.co/konverner/lstm_sentiment models/lstm_sentiment

echo "Model files have been successfully downloaded to the models/lstm_sentiment directory."
