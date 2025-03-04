#!/bin/bash

# Fetch script with argument support
if [[ -z "$1" ]]; then
    read -p "Please enter the path to your sample text file: " sample
else
    sample="$1"
fi

# Check if the file exists
if [[ ! -f "$sample" ]]; then
    echo "Error: File '$sample' not found. Please check the file path."
    exit 1
fi

fetch.py --input "$sample" --where Yolo

