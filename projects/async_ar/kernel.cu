#include "./consumer.cu"
#include "./producer.cu"

#include <rccl/rccl.h>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>


#include <torch/extension.h>

template<int B_LANES, int A_PRODUCERS, int B_PRODUCERS, int CONSUMERS, int QSIZE>
void __global__ _tsr_kernel(
    const fp8* __restrict__ A, 
    const fp8* __restrict__ B,
    half* __restrict__ D,
    const float* scale_tensor,
    const int m,
    const int n,
    const int k,
    const int split_k
) {
    // Initialize shared queue
    __shared__ uint8 queue[2 * B_LANES * QSIZE];
    if (threadIdx.x < 2 * B_LANES * QSIZE) {
        queue[threadIdx.x] = 0;
    }
    // Declare shared buffer
    __shared__ fp8 A_buffer[WARPTILE_M * WARPTILE_K * QSIZE];
    __shared__ fp8 B_buffer[(OP_N * B_LANES) * WARPTILE_K * QSIZE];
    __syncthreads();


    // Infer index and p-state
    int role_id;
    int index;
    uint8 p_state;

    // A producer warp
    if (threadIdx.x < A_PRODUCERS * WARPSIZE) {
        role_id = threadIdx.x / WARPSIZE;
        index = (OPS == 1 ? 2 : 1) * role_id;
        p_state = 0;
    } 
    // B producer warp
    else if (threadIdx.x < A_PRODUCERS * WARPSIZE + B_PRODUCERS * WARPSIZE) {
        role_id = (threadIdx.x / WARPSIZE) - A_PRODUCERS;
        index = role_id;
        p_state = 0;
    }
    // Consumers warp
    else {
        role_id = (threadIdx.x / WARPSIZE) - (A_PRODUCERS + B_PRODUCERS);
        index = role_id;
        p_state = 32;
    }

    // Tiles loop
    int curr_n, curr_k, k_blocks, dropped_rows, dropped_cols;
    const int warptile_per_row = CDIV(n, (OP_N * B_LANES));
    const int tiles = warptile_per_row * split_k;
    const int tpw = max(CDIV(tiles, CU), 1);

    for (int warptile = (tpw * blockIdx.x); warptile < min(tiles, tpw * (blockIdx.x + 1)); warptile++) {

        // Compute tile position
        curr_n = (warptile % warptile_per_row) * (OP_N * B_LANES);
        curr_k = (warptile / warptile_per_row) * WARPTILE_K * K_BLOCKS(k, split_k);
        k_blocks = ((warptile / warptile_per_row) == (split_k - 1)) ? (k / WARPTILE_K) - (split_k - 1) * K_BLOCKS(k, split_k) : K_BLOCKS(k, split_k);

        // Account for column overflow
        dropped_rows = max(0, 0      + WARPTILE_M - m);
        dropped_cols = max(0, curr_n + (OP_N * B_LANES) - n);
        curr_n -= dropped_cols;

        // A producer warp
        if (threadIdx.x < A_PRODUCERS * WARPSIZE) {
            _tsr_A_producer<A_PRODUCERS, B_LANES, QSIZE>(
                A + curr_k, 
                &A_buffer[0], 
                &queue[0],
                index, p_state, role_id,
                dropped_rows,
                k, k_blocks
            ); 
        } 
        // B producer warp
        else if (threadIdx.x < A_PRODUCERS * WARPSIZE + B_PRODUCERS * WARPSIZE) {
            _tsr_B_producer<B_PRODUCERS, B_LANES, QSIZE>(
                B + curr_n * k + curr_k,
                &B_buffer[0],
                &queue[1],
                index, p_state, role_id,
                k, k_blocks
            ); 
        }
        // Consumers warp
        else if (threadIdx.x < (A_PRODUCERS + B_PRODUCERS + CONSUMERS) * WARPSIZE) {
            _tsr_consumer<CONSUMERS, B_LANES, QSIZE>(
                &A_buffer[0],
                &B_buffer[0],
                D + curr_n,
                scale_tensor[0],
                &queue[0],
                index, p_state, role_id,
                n, 
                dropped_rows, dropped_cols,
                k, k_blocks
            );
        }
    }    
}

template<int B_LANES, int A_PRODUCERS, int B_PRODUCERS, int CONSUMERS, int QSIZE>
void __global__ _tsr_kernel_fused(
    const fp8* __restrict__ A, 
    const fp8* __restrict__ B,
    half* __restrict__ D,
    const float* scale_tensor,
    const int m,
    const int n,
    const int k,
    const int split_k,
    half* partial_results,  // Buffer for partial results
    int* ready_flags        // Flags to signal when partial results are ready
) {
    // Initialize shared queue
    __shared__ uint8 queue[2 * B_LANES * QSIZE];
    if (threadIdx.x < 2 * B_LANES * QSIZE) {
        queue[threadIdx.x] = 0;
    }
    __syncthreads();

    // Declare shared buffer
    __shared__ fp8 A_buffer[WARPTILE_M * WARPTILE_K * QSIZE];
    __shared__ fp8 B_buffer[(OP_N * B_LANES) * WARPTILE_K * QSIZE];
    __syncthreads();

    // Infer index and p-state
    int role_id;
    int index;
    uint8 p_state;

    // A producer warp
    if (threadIdx.x < A_PRODUCERS * WARPSIZE) {
        role_id = threadIdx.x / WARPSIZE;
        index = (OPS == 1 ? 2 : 1) * role_id;
        p_state = 0;
    } 
    // B producer warp
    else if (threadIdx.x < A_PRODUCERS * WARPSIZE + B_PRODUCERS * WARPSIZE) {
        role_id = (threadIdx.x / WARPSIZE) - A_PRODUCERS;
        index = role_id;
        p_state = 0;
    }
    // Consumers warp
    else {
        role_id = (threadIdx.x / WARPSIZE) - (A_PRODUCERS + B_PRODUCERS);
        index = role_id;
        p_state = 32;
    }

    // Tiles loop
    int curr_n, curr_k, k_blocks, dropped_rows, dropped_cols;
    const int warptile_per_row = CDIV(n, (OP_N * B_LANES));
    const int tiles = warptile_per_row * split_k;
    const int tpw = max(CDIV(tiles, CU), 1);

    for (int warptile = (tpw * blockIdx.x); warptile < min(tiles, tpw * (blockIdx.x + 1)); warptile++) {

        // Compute tile position
        curr_n = (warptile % warptile_per_row) * (OP_N * B_LANES);
        curr_k = (warptile / warptile_per_row) * WARPTILE_K * K_BLOCKS(k, split_k);
        k_blocks = ((warptile / warptile_per_row) == (split_k - 1)) ? (k / WARPTILE_K) - (split_k - 1) * K_BLOCKS(k, split_k) : K_BLOCKS(k, split_k);

        // Account for column overflow
        dropped_rows = max(0, 0      + WARPTILE_M - m);
        dropped_cols = max(0, curr_n + (OP_N * B_LANES) - n);
        curr_n -= dropped_cols;

        // A producer warp
        if (threadIdx.x < A_PRODUCERS * WARPSIZE) {
            _tsr_A_producer<A_PRODUCERS, B_LANES, QSIZE>(
                A + curr_k, 
                &A_buffer[0], 
                &queue[0],
                index, p_state, role_id,
                dropped_rows,
                k, k_blocks
            ); 
        } 
        // B producer warp
        else if (threadIdx.x < A_PRODUCERS * WARPSIZE + B_PRODUCERS * WARPSIZE) {
            _tsr_B_producer<B_PRODUCERS, B_LANES, QSIZE>(
                B + curr_n * k + curr_k,
                &B_buffer[0],
                &queue[1],
                index, p_state, role_id,
                k, k_blocks
            ); 
        }
        // Consumers warp
        else if (threadIdx.x < (A_PRODUCERS + B_PRODUCERS + CONSUMERS) * WARPSIZE) {
            _tsr_consumer<CONSUMERS, B_LANES, QSIZE>(
                &A_buffer[0],
                &B_buffer[0],
                D + curr_n,
                scale_tensor[0],
                &queue[0],
                index, p_state, role_id,
                n, 
                dropped_rows, dropped_cols,
                k, k_blocks
                //partial_results,  // Pass partial_results buffer
                //ready_flags      // Pass ready_flags buffer
            );
        }
    }
}

// Kernel to sum split-k pieces
__global__ void sum_split_k_kernel(
    half* __restrict__ output,
    const half* __restrict__ pieces,
    size_t elements_per_split,
    int num_pieces
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elements_per_split) {
        half sum = __float2half(0.0f);
        for (int i = 0; i < num_pieces; i++) {
            sum = __hadd(sum, pieces[i * elements_per_split + idx]);
        }
        output[idx] = sum;
    }
}

void skinny_gemm(
    torch::Tensor& A,
    torch::Tensor& B,
    torch::Tensor& D,
    torch::Tensor& scale_tensor,
    int64_t b_lanes,
    int64_t split_k
) {
    const int m = A.size(0);
    const int n = B.size(1);
    const int k = A.size(1);
    
    const fp8* __restrict__ A_ = (const fp8* __restrict__) A.data_ptr(); 
    const fp8* __restrict__ B_ = (const fp8* __restrict__) B.data_ptr(); 
    half* __restrict__ D_ = (half* __restrict__) D.data_ptr(); 
    float* __restrict__ scale_tensor_ = (float* __restrict__) scale_tensor.data_ptr(); 

    // Check shape
    if (m > WARPTILE_M) {
        std::cerr << "m = " << k << " is greater than WARPTILE_M = " << WARPTILE_M << std::endl;
        exit(1);
    }    
    if (k % WARPTILE_K != 0) {
        std::cerr << "k = " << k << " is not divisible by WARPTILE_K = " << WARPTILE_K << std::endl;
        exit(1);
    }

    // Prepare kernel launch
    dim3 grid(CU, 1, 1);
    dim3 block(1, 1, 1);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch kernel (branched on B_LANES)
    switch (b_lanes) {
        case 2:
            block.x = WARPSIZE * (4 + 8 + 4);
            _tsr_kernel<2, 4, 8, 4, 4><<<grid, block, 0, stream>>>(A_, B_, D_, scale_tensor_, m, n, k, split_k);
            break;
        case 3:
            block.x = WARPSIZE * (2 + 6 + 3);
            _tsr_kernel<3, 2, 6, 3, 3><<<grid, block, 0, stream>>>(A_, B_, D_, scale_tensor_, m, n, k, split_k);
            break;
        case 4:
            block.x = WARPSIZE * (2 + 6 + 3);
            _tsr_kernel<4, 2, 6, 3, 3><<<grid, block, 0, stream>>>(A_, B_, D_, scale_tensor_, m, n, k, split_k);
            break;
        case 5:
            block.x = WARPSIZE * (2 + 9 + 2);
            _tsr_kernel<5, 2, 9, 2, 2><<<grid, block, 0, stream>>>(A_, B_, D_, scale_tensor_, m, n, k, split_k);
            break;
        default:
            break;
    }
}

void skinny_gemm_ar(
    torch::Tensor& A,
    torch::Tensor& B,
    torch::Tensor& D,
    torch::Tensor& scale_tensor,
    int64_t b_lanes,
    int64_t split_k,
    ncclComm_t comm
) {
    const int m = A.size(0);
    const int n = B.size(1);
    const int k = A.size(1);
    
    const fp8* __restrict__ A_ = (const fp8* __restrict__) A.data_ptr(); 
    const fp8* __restrict__ B_ = (const fp8* __restrict__) B.data_ptr(); 
    half* __restrict__ D_ = (half* __restrict__) D.data_ptr(); 
    float* __restrict__ scale_tensor_ = (float* __restrict__) scale_tensor.data_ptr(); 

    int rank, size;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &size);
    printf("Rank %d: Starting kernel with size=%d\n", rank, size);

    // Prepare kernel launch
    dim3 grid(CU, 1, 1);
    dim3 block(1, 1, 1);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Calculate sizes
    const size_t elements_per_split = m * n;
    const size_t partial_size = elements_per_split * sizeof(half);
    printf("Rank %d: elements_per_split=%zu, partial_size=%zu\n", rank, elements_per_split, partial_size);
    
    // Allocate memory
    half* partial_results = nullptr;
    half* received_pieces = nullptr;
    int* ready_flags = nullptr;
    hipMalloc(&partial_results, split_k * partial_size);
    hipMalloc(&received_pieces, size * partial_size);
    hipMalloc(&ready_flags, split_k * sizeof(int));
    hipMemsetAsync(ready_flags, 0, split_k * sizeof(int), stream);
    hipMemsetAsync(partial_results, 0, split_k * partial_size, stream);
    hipMemsetAsync(received_pieces, 0, size * partial_size, stream);
    printf("Rank %d: Memory allocated and initialized\n", rank);

    // Create communication stream
    cudaStream_t comm_stream;
    hipStreamCreate(&comm_stream);
    printf("Rank %d: Communication stream created\n", rank);

    // Launch kernel
    printf("Rank %d: Launching kernel with b_lanes=%ld\n", rank, b_lanes);
    switch (b_lanes) {
        case 2:
            block.x = WARPSIZE * (4 + 8 + 4);
            _tsr_kernel_fused<2, 4, 8, 4, 4><<<grid, block, 0, stream>>>(
                A_, B_, D_, scale_tensor_, m, n, k, split_k, partial_results, ready_flags);
            break;
        case 3:
            block.x = WARPSIZE * (2 + 6 + 3);
            _tsr_kernel_fused<3, 2, 6, 3, 3><<<grid, block, 0, stream>>>(
                A_, B_, D_, scale_tensor_, m, n, k, split_k, partial_results, ready_flags);
            break;
        case 4:
            block.x = WARPSIZE * (2 + 6 + 3);
            _tsr_kernel_fused<4, 2, 6, 3, 3><<<grid, block, 0, stream>>>(
                A_, B_, D_, scale_tensor_, m, n, k, split_k, partial_results, ready_flags);
            break;
        case 5:
            block.x = WARPSIZE * (2 + 9 + 2);
            _tsr_kernel_fused<5, 2, 9, 2, 2><<<grid, block, 0, stream>>>(
                A_, B_, D_, scale_tensor_, m, n, k, split_k, partial_results, ready_flags);
            break;
        default:
            break;
    }
    printf("Rank %d: Kernel launched\n", rank);

    // Track completion
    std::vector<int> host_flags(split_k, 0);
    std::vector<int> processed_flags(split_k, 0);
    int completed = 0;

    printf("Rank %d: Entering completion loop\n", rank);
    while (completed < split_k) {
        // Copy flags to host
        //hipMemcpy(host_flags.data(), ready_flags, split_k * sizeof(int), hipMemcpyDeviceToHost);
        
        for (int i = 0; i < split_k; i++) {
            if (host_flags[i] == 1 && !processed_flags[i]) {
                printf("Rank %d: Processing split %d\n", rank, i);
                size_t offset = i * elements_per_split;
                
                // Broadcast ready state
                int ready = 1;
                printf("Rank %d: Broadcasting ready state for split %d\n", rank, i);
                ncclGroupStart();
                for (int src = 0; src < size; src++) {
                    ncclBroadcast(
                        &ready,
                        &ready,
                        1,
                        ncclInt,
                        src,
                        comm,
                        comm_stream
                    );
                }
                ncclGroupEnd();
                printf("Rank %d: Ready state broadcast complete for split %d\n", rank, i);

                // Broadcast data
                printf("Rank %d: Broadcasting data for split %d\n", rank, i);
                ncclGroupStart();
                for (int src = 0; src < size; src++) {
                    ncclBroadcast(
                        (src == rank) ? (partial_results + offset) : nullptr,
                        received_pieces + (src * elements_per_split),
                        elements_per_split,
                        ncclHalf,
                        src,
                        comm,
                        comm_stream
                    );
                }
                ncclGroupEnd();
                printf("Rank %d: Data broadcast complete for split %d\n", rank, i);
                hipStreamSynchronize(comm_stream);

                // Sum pieces
                printf("Rank %d: Launching reduction kernel for split %d\n", rank, i);
                dim3 sum_grid((elements_per_split + 255) / 256);
                dim3 sum_block(256);
                hipLaunchKernelGGL(sum_split_k_kernel,
                    sum_grid,
                    sum_block,
                    0,
                    stream,
                    D_ + offset,
                    received_pieces,
                    elements_per_split,
                    size
                );
                printf("Rank %d: Reduction complete for split %d\n", rank, i);

                processed_flags[i] = 1;
                completed++;
            }
        }

        if (completed < split_k) {
            printf("Rank %d: Waiting... completed=%d/%d\n", rank, completed, split_k);
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }

    printf("Rank %d: All splits completed\n", rank);
    hipStreamSynchronize(comm_stream);
    hipStreamSynchronize(stream);

    // Cleanup
    hipStreamDestroy(comm_stream);
    hipFree(partial_results);
    hipFree(received_pieces);
    hipFree(ready_flags);
    printf("Rank %d: Cleanup complete\n", rank);
}

// Initialize NCCL and return the communicator
ncclComm_t initialize_nccl(int8_t world_size, int8_t rank) {
    ncclUniqueId unique_id;
    ncclComm_t comm;

    // Generate unique ID on rank 0 and broadcast it to all ranks
    if (rank == 0) {
        ncclGetUniqueId(&unique_id);
    }

    // Broadcast unique ID to all ranks (assuming MPI or similar)
    // For simplicity, this example assumes rank 0 initializes and broadcasts the ID
    // In a real distributed setup, you would use MPI or another method to broadcast the ID

    // Initialize NCCL communicator
    cudaSetDevice(rank);
    ncclCommInitRank(&comm, world_size, unique_id, rank);

    return comm;
}

class FusedGEMMAR {
private:
    int world_size;
    int rank;
    ncclUniqueId nccl_id;
    ncclComm_t comm;
    
    void init_nccl(c10d::ProcessGroup& pg) {
        if (rank == 0) {
            ncclGetUniqueId(&nccl_id);
            auto nccl_tensor = torch::from_blob(&nccl_id, {sizeof(ncclUniqueId)}, torch::kByte);
            std::vector<torch::Tensor> tensors = {nccl_tensor};
            pg.broadcast(tensors)->wait();
        } else {
            auto nccl_tensor = torch::empty({sizeof(ncclUniqueId)}, torch::TensorOptions().dtype(torch::kByte));
            std::vector<torch::Tensor> tensors = {nccl_tensor};
            pg.broadcast(tensors)->wait();
            memcpy(&nccl_id, nccl_tensor.data_ptr(), sizeof(ncclUniqueId));
        }
        ncclCommInitRank(&comm, world_size, nccl_id, rank);
    }

public:
    FusedGEMMAR(int world_size_, int rank_, c10d::ProcessGroup& process_group) 
        : world_size(world_size_), 
          rank(rank_) {
        init_nccl(process_group);
    }

    ~FusedGEMMAR() {
        ncclCommDestroy(comm);
    }

    void gemm_ar(
        torch::Tensor& A,
        torch::Tensor& B,
        torch::Tensor& D,
        torch::Tensor& scale_tensor,
        int64_t b_lanes,
        int64_t split_k
    ) {
        skinny_gemm_ar(A, B, D, scale_tensor, b_lanes, split_k, comm);
        //skinny_gemm(A, B, D, scale_tensor, b_lanes, split_k);
    }
};


#define PYBIND11_MODULE_EXPAND(NAME, MODULE) PYBIND11_MODULE(NAME, MODULE)

PYBIND11_MODULE_EXPAND(TORCH_EXTENSION_NAME, m) {
    py::class_<FusedGEMMAR>(m, "FusedGEMMAR")
        .def(py::init<int, int, c10d::ProcessGroup&>())
        .def("gemm_ar", &FusedGEMMAR::gemm_ar);
}