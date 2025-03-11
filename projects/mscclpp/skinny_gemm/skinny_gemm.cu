#include "./consumer.cu"
#include "./producer.cu"
#include <mscclpp/concurrency_device.hpp>


template <class T>
using DeviceHandle = mscclpp::DeviceHandle<T>;
__constant__ DeviceHandle<mscclpp::PortChannel> constRingChannels[7];

__device__ mscclpp::DeviceSyncer deviceSyncer;

__global__ void vectorized_half_sum_inplace(__half* __restrict__ D, const __half* __restrict__ input_buff, int size, int rank, int world_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    using half2_t = __half2;

    // Ring all-reduce
    for (int step = 0; step < world_size - 1; ++step) {
        for (int i = idx; i < size / 2; i += stride) {
            half2_t a = reinterpret_cast<half2_t*>(D)[i];
            half2_t b = reinterpret_cast<const half2_t*>(input_buff)[i];
            reinterpret_cast<half2_t*>(D)[i] = __hadd2(a, b);  // In-place addition
        }
    
        // Handle odd-length case (if n is odd)
        if (idx == 0 && (size % 2) != 0) {
            D[size - 1] = __hadd(D[size - 1], input_buff[size - 1]);
        }

        // kick data around the ring
        if (idx == 0) {
            int peerSendRank = (rank + 1) % world_size;
            int peerRecvRank = (rank - 1 + world_size) % world_size;
            int peerSendId = peerSendRank < rank ? peerSendRank : peerSendRank - 1;
            int peerRecvId = peerRecvRank < rank ? peerRecvRank : peerRecvRank - 1;
            DeviceHandle<mscclpp::PortChannel>& left = constRingChannels[peerRecvId];
            DeviceHandle<mscclpp::PortChannel>& right = constRingChannels[peerSendId];
            printf("Allreduce Rank %d: Sending data to %d\n", rank, peerSendRank);
            right.putWithSignal(0, size);
            right.flush();
            left.wait();
            printf("Allreduce Rank %d: Received data from %d\n", rank, peerRecvRank);
            __syncthreads();
        }

        __syncthreads(); // Ensure all threads are synchronized after communication
    }
}

#define launch_tsr(BL, AP, BP, C, QS)                                                                        \
    block.x = WARPSIZE * (AP + BP + C); \
    _tsr_kernel<BL, AP, BP, C, QS><<<grid, block, 0, stream>>>(A_, B_, D_, scale_tensor_, m, n, k, split_k, rank, world_size, scratch_); \
    break;

template <int B_LANES, int A_PRODUCERS, int B_PRODUCERS, int CONSUMERS, int QSIZE>
void __global__ _tsr_kernel(const fp8* __restrict__ A, const fp8* __restrict__ B, half* __restrict__ D,
                            const float* scale_tensor, const int m, const int n, const int k, const int split_k, const int rank, const int world_size, half* scratch) {
    // Initialize shared queue
    __shared__ int queue[2 * B_LANES * QSIZE];
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
    int p_state;

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
        p_state = 1;
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
        k_blocks = ((warptile / warptile_per_row) == (split_k - 1))
                       ? (k / WARPTILE_K) - (split_k - 1) * K_BLOCKS(k, split_k)
                       : K_BLOCKS(k, split_k);

        // Account for column overflow
        dropped_rows = max(0, 0 + WARPTILE_M - m);
        dropped_cols = max(0, curr_n + (OP_N * B_LANES) - n);
        curr_n -= dropped_cols;

        // A producer warp
        if (threadIdx.x < A_PRODUCERS * WARPSIZE) {
            _tsr_A_producer<A_PRODUCERS, B_LANES, QSIZE>(A + curr_k, &A_buffer[0], &queue[0], index, p_state, role_id,
                                                         dropped_rows, k, k_blocks);
        }
        // B producer warp
        else if (threadIdx.x < A_PRODUCERS * WARPSIZE + B_PRODUCERS * WARPSIZE) {
            _tsr_B_producer<B_PRODUCERS, B_LANES, QSIZE>(B + curr_n * k + curr_k, &B_buffer[0], &queue[1], index,
                                                         p_state, role_id, k, k_blocks);
        }
        // Consumers warp
        else if (threadIdx.x < (A_PRODUCERS + B_PRODUCERS + CONSUMERS) * WARPSIZE) {
            //printf("Consumer starting\n");
            _tsr_consumer<CONSUMERS, B_LANES, QSIZE>(&A_buffer[0], &B_buffer[0], D + curr_n, scale_tensor[0], &queue[0],
                                                     index, p_state, role_id, n, dropped_rows, dropped_cols, k,
                                                     k_blocks, scratch);

            if (threadIdx.x == (A_PRODUCERS + B_PRODUCERS) * WARPSIZE) {
                // Send th result around the ring, only one threads needs to do this
                int peerSendRank = (rank + 1) % world_size;
                int peerRecvRank = (rank - 1 + world_size) % world_size;
                int peerSendId = peerSendRank < rank ? peerSendRank : peerSendRank - 1;
                int peerRecvId = peerRecvRank < rank ? peerRecvRank : peerRecvRank - 1;
                DeviceHandle<mscclpp::PortChannel>& left = constRingChannels[peerRecvId];
                DeviceHandle<mscclpp::PortChannel>& right = constRingChannels[peerSendId];
                printf("Rank %d: Sending data to %d\n", rank, peerSendRank);
                right.putWithSignal(curr_n, WARPTILE_M * (OP_N * B_LANES) * 2);
                right.flush();
                left.wait();
                printf("Rank %d: Received data from %d\n", rank, peerRecvRank);
                deviceSyncer.sync(gridDim.x);
                __syncthreads();
            }
        }
    }
}

void skinny_gemm(torch::Tensor& A, torch::Tensor& B, torch::Tensor& D, torch::Tensor& scale_tensor, int64_t b_lanes,
                 int64_t split_k, const int rank, const int world_size, uint8_t* scratch) {
    const int m = A.size(0);
    const int n = B.size(1);
    const int k = A.size(1);

    const fp8* __restrict__ A_ = (const fp8* __restrict__)A.data_ptr();
    const fp8* __restrict__ B_ = (const fp8* __restrict__)B.data_ptr();
    half* __restrict__ D_ = (half* __restrict__)D.data_ptr();
    float* __restrict__ scale_tensor_ = (float* __restrict__)scale_tensor.data_ptr();

    half* __restrict__ scratch_ = (half* __restrict__)scratch;

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
            launch_tsr(2, 3, 8, 4, 5);
        case 3:
            launch_tsr(3, 3, 5, 2, 4);  // Perforamnce on MI300: 8_13312_16384:57.54
        case 4:
            launch_tsr(4, 2, 6, 3, 3);  // Perforamnce on MI300: 8_16384_6656:29.5
        case 5:
            launch_tsr(5, 2, 6, 2, 2);
        default:
            break;
    }

    //Reduction
    int threads = 256;
    int blocks = (D.numel() / 2 + threads - 1) / threads;
    vectorized_half_sum_inplace<<<blocks, threads, 0, stream>>>(D_, scratch_, D.numel(), rank, world_size);

    cudaStreamSynchronize(stream);
}