#include <hip/hip_runtime.h>
#include <mscclpp/port_channel.hpp>

// Device-side kernel for local reduction and communication
__global__ void allReduceKernel(float* data, size_t count,
                               mscclpp::DeviceHandle<mscclpp::PortChannel>* channels,
                               int numChannels) {
    // Each thread handles a portion of the data
    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; 
         i < count; 
         i += blockDim.x * gridDim.x) {
        
        float sum = data[i];
        
        // Ring allreduce algorithm
        for (int step = 0; step < numChannels; step++) {
            // Only one thread per block handles communication
            if (threadIdx.x == 0) {
                size_t dataSize = sizeof(float);
                uint64_t offset = i * sizeof(float);
                
                // Send my data to next rank
                channels[step].putWithSignal(offset, offset, dataSize);
                
                // Wait for completion
                channels[step].wait();
            }
            __syncthreads();
        }
        
        data[i] = sum;
    }
}

void launch_allreduce(float* data, size_t count,
                     mscclpp::DeviceHandle<mscclpp::PortChannel>* channels,
                     int numChannels, hipStream_t stream) {
    const int blockSize = 256;
    const int numBlocks = (count + blockSize - 1) / blockSize;
    
    hipLaunchKernelGGL(allReduceKernel, dim3(numBlocks), dim3(blockSize), 
                       0, stream, data, count, channels, numChannels);
} 