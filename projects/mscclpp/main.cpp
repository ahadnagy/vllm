int main(int argc, char** argv) {
    // Initialize MPI or other process management
    int rank = 0;  // Get rank from your process management system
    int worldSize = 4;  // Total number of GPUs
    
    // Set device
    cudaSetDevice(rank);
    
    // Allocate and initialize data
    const size_t count = 1024;
    float* data;
    cudaMalloc(&data, count * sizeof(float));
    
    // Initialize data on each GPU
    // ... (initialization code here)
    
    // Create AllReduce instance
    AllReduce allReduce(rank, worldSize);
    
    // Perform all-reduce
    allReduce.reduce(data, count);
    
    // Cleanup
    cudaFree(data);
    
    return 0;
} 