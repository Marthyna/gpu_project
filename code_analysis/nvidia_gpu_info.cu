#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "    Shared Memory Per Block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
        std::cout << "    Streaming Multiprocessors: " << prop.multiProcessorCount << std::endl;
    }
    return 0;
}
