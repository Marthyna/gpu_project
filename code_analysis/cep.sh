#!/bin/bash

# Check if a program name has been passed as a command-line argument
if [ $# -eq 0 ]; then
    cuda_program_name="stem_conv"
else
    cuda_program_name="$1"
fi

# Compile the CUDA program
nvcc -o "./$cuda_program_name" "./$cuda_program_name.cu" device_operations.cpp

# Check if the compilation was successful
if [ $? -eq 0 ]; then
    # Execute the compiled program
    "./$cuda_program_name"
    "./$cuda_program_name"

    # Profile the execution using nvprof
    nvprof "./$cuda_program_name"
else
    echo "Error during compilation of the CUDA program."
fi
