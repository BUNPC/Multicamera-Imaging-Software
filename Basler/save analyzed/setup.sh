#!/bin/bash

# List of Python modules to install
modules=("pypylon" "numpy" "h5py")

# Loop through the list and install each module
for module in "${modules[@]}"; do
    pip install "$module"
    if [ $? -eq 0 ]; then
        echo "Successfully installed $module"
    else
        echo "Failed to install $module"
    fi
done