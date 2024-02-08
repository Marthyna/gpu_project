CC=nvcc
OUT_NAME=matrix_gpu
GPU=vmgpu034

build: clean
	scp matrix_gpu.cu weberml@$(GPU).ensimag.fr:~/lab1
	ssh weberml@$(GPU).ensimag.fr "cd lab1; /usr/local/cuda/bin/nvcc matrix_gpu.cu -o $(OUT_NAME)"
	ssh weberml@$(GPU).ensimag.fr "cd lab1;./$(OUT_NAME)"

.PHONY: clean build 