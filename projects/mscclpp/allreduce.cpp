#include <mscclpp/communicator.hpp>
#include <mscclpp/proxy_service.hpp>
#include <mscclpp/port_channel.hpp>
#include <cuda_runtime.h>

// Device-side kernel for local reduction and communication
__global__ void allReduceKernel(float* data, size_t count, 
                               mscclpp::DeviceHandle<mscclpp::PortChannel>* channels,
                               int numChannels) {
    // Each GPU first sends its data to the next GPU in ring
    for (int step = 0; step < numChannels; step++) {
        // Only one thread handles communication
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            // Send data to next GPU
            channels[step].put(/*dstOffset=*/0, /*srcOffset=*/0, 
                             count * sizeof(float));
            channels[step].flush();
            
            // Wait for data from previous GPU
            channels[step].wait();
        }
        
        // Synchronize all threads
        __syncthreads();
        
        // Perform local reduction
        for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; 
             i < count; 
             i += blockDim.x * gridDim.x) {
            atomicAdd(&data[i], data[i]);
        }
    }
}

class AllReduce {
public:
    AllReduce(int rank, int worldSize) : rank_(rank), worldSize_(worldSize) {
        // Initialize bootstrap for connection setup
        bootstrap_ = std::make_shared<mscclpp::TcpBootstrap>(rank, worldSize);
        comm_ = std::make_unique<mscclpp::Communicator>(bootstrap_);
        
        // Setup channels for ring communication
        setupChannels();
        
        // Initialize proxy service
        proxyService_ = std::make_unique<mscclpp::ProxyService>();
    }

    void reduce(float* data, size_t count) {
        // Start proxy service
        proxyService_->startProxy();
        
        // Launch kernel for reduction
        const int blockSize = 256;
        const int numBlocks = (count + blockSize - 1) / blockSize;
        
        allReduceKernel<<<numBlocks, blockSize>>>(
            data, count, deviceChannels_.get(), worldSize_ - 1);
        
        // Stop proxy service
        proxyService_->stopProxy();
    }

private:
    void setupChannels() {
        // Create channels in ring topology
        channels_.resize(worldSize_ - 1);
        
        // Setup connections with next and previous ranks
        for (int i = 0; i < worldSize_ - 1; i++) {
            int nextRank = (rank_ + 1) % worldSize_;
            channels_[i] = std::make_unique<mscclpp::PortChannel>(
                *comm_, rank_, nextRank);
        }
        
        // Copy channels to device
        cudaMalloc(&deviceChannels_, 
                   (worldSize_ - 1) * sizeof(mscclpp::DeviceHandle<mscclpp::PortChannel>));
        for (int i = 0; i < worldSize_ - 1; i++) {
            channels_[i]->copyToDevice(&deviceChannels_[i]);
        }
    }

    int rank_;
    int worldSize_;
    std::shared_ptr<mscclpp::TcpBootstrap> bootstrap_;
    std::unique_ptr<mscclpp::Communicator> comm_;
    std::unique_ptr<mscclpp::ProxyService> proxyService_;
    std::vector<std::unique_ptr<mscclpp::PortChannel>> channels_;
    mscclpp::DeviceHandle<mscclpp::PortChannel>* deviceChannels_;
}; 