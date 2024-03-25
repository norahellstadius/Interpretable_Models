#!/bin/bash

# Path to the base directory
BASE_PATH="/Users/norahallqvist/Code/Interpretable_Models/tests"

# Check if the base directory exists
if [ ! -d "$BASE_PATH" ]; then
    echo "Error: Base directory does not exist."
    exit 1
fi

# Find all Python files in the base directory and its subdirectories starting with "test_"
find "$BASE_PATH" -type f -name 'test_*.py' -print |
while read -r file; do
    echo "Running file: $file"
    # Run the Python file using python3
    python3 "$file"
done
