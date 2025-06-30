#!/bin/bash

# List of scenes to process
scenes=("bicycle" "bonsai" "counter" "garden" "kitchen" "room" "stump")

# Path for storing method results
method_path="results"

# Directory containing images
image_dir="images_4"

# Port number for the process
port=1225

# Initialization method:
# 0 - Original 3DGS initialization
# 1 - SLV random initialization
random=1

# Method selection:
# 1 - Use our method
# 0 - Use original 3DGS
ours=1

# Iterate through each scene and run training
for scene in "${scenes[@]}"; do
    python train.py \
        -s "../Data/${scene}/" \
        --eval \
        -i "$image_dir" \
        -m "./output/${scene}/${method_path}" \
        --port "$port" \
        --random "$random" \
        --ours "$ours"
done