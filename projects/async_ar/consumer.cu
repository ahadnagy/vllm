#include "./core.cu"

void inline __device__ consumer_smem_to_reg8(fp8* buffer, fp8x8 &reg) 
{
    // 32 bits load from the current bank
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        reg[i] = buffer[i];
    }
    // 32 bits load from the same bank, hopefully extension of the first load
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        reg[4 + i] = buffer[i + 32*E_P_BANK];
    }
}

void inline __device__ consumer_smem_to_reg16(fp8* buffer, fp8x16 &reg) 
{
    #pragma unroll
    for (int i = 0; i < 4; i++) { reg[i     ] = buffer[i                   ]; }
    #pragma unroll
    for (int i = 0; i < 4; i++) { reg[i +  4] = buffer[i +     32*E_P_BANK]; }
    #pragma unroll
    for (int i = 0; i < 4; i++) { reg[i +  8] = buffer[i + 2 * 32*E_P_BANK]; }
    #pragma unroll
    for (int i = 0; i < 4; i++) { reg[i + 12] = buffer[i + 3 * 32*E_P_BANK]; }
}

// Helper function to convert fp8 to float
__device__ float fp8_to_float(fp8 val) {
    return static_cast<float>(val) * (1.0f / 256.0f); // Adjust scaling as needed
}

// Helper function to convert fp8 to __half
__device__ __half fp8_to_half(fp8 val) {
    return __float2half(fp8_to_float(val)); // Convert fp8 to float, then to __half
}

// Convert fp8x8 to __half2
__device__ __half2 fp8x8_to_half2(const fp8x8 &vec, int index) {
    __half2 result;
    result.x = fp8_to_half(vec[index * 2]);     // First fp8 value
    result.y = fp8_to_half(vec[index * 2 + 1]); // Second fp8 value
    return result;
}

// Convert fp8x16 to __half2
__device__ __half2 fp8x16_to_half2(const fp8x16 &vec, int index) {
    __half2 result;
    result.x = fp8_to_half(vec[index * 2]);     // First fp8 value
    result.y = fp8_to_half(vec[index * 2 + 1]); // Second fp8 value
    return result;
}

// Convert fp8x8 to float2
__device__ float2 fp8x8_to_float2(const fp8x8 &vec, int index) {
    float2 result;
    result.x = fp8_to_float(vec[index * 2]);     // First fp8 value
    result.y = fp8_to_float(vec[index * 2 + 1]); // Second fp8 value
    return result;
}

// Convert fp8x16 to float2
__device__ float2 fp8x16_to_float2(const fp8x16 &vec, int index) {
    float2 result;
    result.x = fp8_to_float(vec[index * 2]);     // First fp8 value
    result.y = fp8_to_float(vec[index * 2 + 1]); // Second fp8 value
    return result;
}

template<int CONSUMERS, int B_LANES, int QSIZE>
void __device__ _tsr_consumer(
    fp8* A_buffer,
    fp8* B_buffer,
    half* D,
    float scale,
    uint8* queue,
    int &index,
    uint8 &p_state,
    int &role_id,
    const int n,
    const int dropped_rows,
    const int dropped_cols,
    const int k,
    const int k_blocks
) {
    // Compute thread position
    const int thread_id = threadIdx.x % WARPSIZE;
    A_buffer += (thread_id / 2) * E_P_BANK + (threadIdx.x % 2) * 32*E_P_BANK * 2;
    B_buffer += (thread_id % 32) * 4 + (thread_id / 32) * 32*E_P_BANK * 4;
    const int sparsity_indices = (threadIdx.x % 2) ? 0x0000EEEE : 0x00004444;

    printf("DEBUG: Consumer thread %d (role_id %d) starting consumer function with initial p_state=%d, index=%d\n", 
           threadIdx.x, role_id, p_state, index);

    // Declare input registers
    fp8x8 reg_A[OPS];
    fp8x16 reg_B[OPS];

    // Initialize output registers
    f32x4 reg_D[B_LANES];
    #pragma unroll
    for (int i = 0; i < (B_LANES); i++) {
        reg_D[i][0] = 0.0f; reg_D[i][1] = 0.0f; reg_D[i][2] = 0.0f; reg_D[i][3] = 0.0f;
    }

    // K-wise loop
    fp8 *A_offs_buff, *B_offs_buff;
    int b = role_id;

    while (b < k_blocks) {
        // // Account for cyclic queue
        // int local_index = index;
        // if (local_index >= QSIZE) {
        //     local_index -= QSIZE;
        //     printf("DEBUG: Consumer thread %d adjusted index from %d to %d\n", 
        //            threadIdx.x, index, local_index);
        // }
        
        // A_offs_buff = A_buffer + local_index * (WARPTILE_M * WARPTILE_K);
        // B_offs_buff = B_buffer + local_index * ((OP_N * B_LANES) * WARPTILE_K);

        // // Wait for A buffer to be filled
        // int queue_idx = 2 * B_LANES * local_index;
        // printf("DEBUG: Consumer thread %d waiting for A buffer. Queue[%d]=%d, p_state=%d\n", 
        //        threadIdx.x, queue_idx, queue[queue_idx], p_state);
        
        // __threadfence();  // Ensure memory consistency
        
        // int wait_count = 0;
        // uint8 expected_state = p_state;
        // while (queue[queue_idx] != expected_state) {
        //     if (wait_count++ % 100000 == 0) {  // Reduced print frequency
        //         printf("DEBUG: Consumer thread %d still waiting for A buffer. Queue[%d]=%d, expecting=%d (wait_count=%d)\n", 
        //                threadIdx.x, queue_idx, queue[queue_idx], expected_state, wait_count);
        //         __threadfence();  // Refresh memory view
        //     }
        //     if (wait_count > 10000000) {  // Add timeout
        //         printf("ERROR: Consumer thread %d timeout waiting for A buffer\n", threadIdx.x);
        //         return;
        //     }
        //     asm volatile("s_sleep 1");  // Increased sleep time slightly
        // }
        
        // printf("DEBUG: Consumer thread %d received A buffer\n", threadIdx.x);

        // // Load A buffer and mark as consumed
        // #pragma unroll
        // for (int op = 0; op < OPS; op++) {
        //     consumer_smem_to_reg8(A_offs_buff + (op * OP_M * OP_K), reg_A[op]);
        // }
        
        // queue[queue_idx] = p_state + 32;
        
        // printf("DEBUG: Consumer thread %d marked A buffer as consumed (Queue[%d]=%d)\n", 
        //        threadIdx.x, queue_idx, queue[queue_idx]);

        // Account for cyclic queue
        index -= (index >= QSIZE) ? QSIZE : 0;
        A_offs_buff = A_buffer + index * (WARPTILE_M * WARPTILE_K);
        B_offs_buff = B_buffer + index * ((OP_N * B_LANES) * WARPTILE_K);

        // Wait for A buffer to be filled
        while (queue[2 * B_LANES * index] != p_state) {
            asm volatile("s_sleep 0");
        }
        // Load A buffer
        #pragma unroll
        for (int op = 0; op < OPS; op++) {
            consumer_smem_to_reg8(A_offs_buff + (op * OP_M * OP_K), reg_A[op]);
        }
        // Mark A buffer as consumed
        queue[2 * B_LANES * index] = p_state + 32;

        // Go through each lane
        #pragma unroll
        for (int lane = 0; lane < B_LANES; lane++) {
            // Wait for B buffer to be filled
            //queue_idx = 2 * (B_LANES * local_index + lane) + 1;
            // printf("DEBUG: Consumer thread %d waiting for B buffer lane %d. Queue[%d]=%d, p_state=%d\n", 
            //        threadIdx.x, lane, queue_idx, queue[queue_idx], p_state);
            
            // __threadfence();  // Ensure memory consistency
            
            // wait_count = 0;
            // while (queue[queue_idx] != p_state) {
            //     if (wait_count++ % 100000 == 0) {
            //         printf("DEBUG: Consumer thread %d waiting for B buffer lane %d. Queue[%d]=%d, p_state=%d (wait_count=%d)\n", 
            //                threadIdx.x, lane, queue_idx, queue[queue_idx], p_state, wait_count);
            //         __threadfence();
            //     }
            //     if (wait_count > 10000000) {
            //         printf("ERROR: Consumer thread %d timeout waiting for B buffer lane %d\n", 
            //                threadIdx.x, lane);
            //         return;
            //     }
            //     asm volatile("s_sleep 1");
            // }
            
            // printf("DEBUG: Consumer thread %d received B buffer for lane %d\n", threadIdx.x, lane);

            // // Load B buffer and mark as consumed
            // #pragma unroll
            // for (int op = 0; op < OPS; op++) {
            //     consumer_smem_to_reg16(B_offs_buff + (lane * OP_N * WARPTILE_K) + (op * OP_N * OP_K), reg_B[op]);
            // }
            
            // __threadfence();
            // queue[queue_idx] = p_state + 32;
            // __threadfence();
            
            // printf("DEBUG: Consumer thread %d marked B buffer lane %d as consumed (Queue[%d]=%d)\n", 
            //        threadIdx.x, lane, queue_idx, queue[queue_idx]);

             // Wait for B buffer to be filled
            while (queue[2 * (B_LANES * index + lane) + 1] != p_state) {
                asm volatile("s_sleep 0");
            }
            // Load B buffer
            #pragma unroll
            for (int op = 0; op < OPS; op++) {
                consumer_smem_to_reg16(B_offs_buff + (lane * OP_N * WARPTILE_K) + (op * OP_N * OP_K), reg_B[op]);
            }
            // Mark B buffer as consumed
            queue[2 * (B_LANES * index + lane) + 1] = p_state + 32;


            // Consume registers
            #pragma unroll
            for (int op = 0; op < OPS; op++) {
                for (int i = 0; i < 16; i++) {
                    for (int j = 0; j < 16; j++) {
                        float sum = 0.0f;
                        #pragma unroll
                        for (int k = 0; k < 16; k++) {
                            // Convert fp8 to float and perform multiplication
                            sum += fp8_to_float(reg_A[op][i * 16 + k]) * fp8_to_float(reg_B[op][k * 16 + j]);
                        }
                        // Accumulate the result
                        reg_D[lane][i * 16 + j] += sum;
                    }
                }
            }
        }

        // Update index and state
        // index += CONSUMERS;
        // if (index >= QSIZE) {
        //     p_state += 64;
        //     printf("DEBUG: Consumer thread %d wrapped queue, new p_state=%d\n", 
        //            threadIdx.x, p_state);
        // }
        // b += CONSUMERS;
        // printf("DEBUG: Consumer thread %d updated index to %d and p_state to %d\n", 
        //        threadIdx.x, index, p_state);

        // Update index
        index += CONSUMERS;
        p_state = (index >= QSIZE) ? p_state + 64 : p_state;
        b += CONSUMERS;
    }

    //printf("DEBUG: Consumer thread %d completed consumer function\n", threadIdx.x);

    // Bring warps back in order
    role_id = b - k_blocks;

    // Infer the current column in D
    int out_n = 2 * ((thread_id % 16) / 2);

    // Finalize registers so that each threads hold 2 consecutive results in memory
    float final_scale;
    int id_to_swap = 1 - threadIdx.x % 2;
    int src_lane = thread_id + 1 - 2 * (thread_id % 2);

    #pragma unroll
    for (int i = 0; i < B_LANES; i++) {

        // Fusing
        reg_D[i][0] = reg_D[i][0] + reg_D[i][1];
        reg_D[i][1] = reg_D[i][2] + reg_D[i][3];

        // Scaling
        final_scale = (out_n + i * OP_N) >= dropped_cols ? scale : 0.0f;
        reg_D[i][0] *= final_scale;
        reg_D[i][1] *= final_scale;

        // Swapping 
        reg_D[i][id_to_swap] = __shfl(reg_D[i][id_to_swap], src_lane);
    }

    // Infer the current row in D
    int out_m = (thread_id / 16) * 2 + (thread_id % 2);

    // If we are in dropped rows territory, we can return now
    if (out_m + dropped_rows > WARPTILE_M -1) {
        return ;
    }

    // Relocate on D
    __half2* D_ = reinterpret_cast<__half2*>(D) + (out_m * n + out_n) / 2;

    // Out lane by lane
    __half2 x;

    #pragma unroll
    for (int i = 0; i < B_LANES; i++) {
        // Form the packed f16
        x.x = __float2half(reg_D[i][0]);
        x.y = __float2half(reg_D[i][1]);

        // Convert __half2 to float2
        float2 x_float2;
        x_float2.x = __half2float(x.x);
        x_float2.y = __half2float(x.y);

        // Perform atomic addition in FP32
        atomicAdd(reinterpret_cast<float*>(&D_[i * OP_N / 2]), x_float2.x);
        atomicAdd(reinterpret_cast<float*>(&D_[i * OP_N / 2]) + 1, x_float2.y);
    }

    // Disabled: if D is of type float
    // // Relocate on D
    // D += (out_m * n + out_n);

    // // Out lane by lane
    // #pragma unroll
    // for (int i = 0; i < B_LANES; i++) {
    //     atomicAdd(&D[0 + i*OP_N], reg_D[i][0]);
    //     atomicAdd(&D[1 + i*OP_N], reg_D[i][1]);
    // }
}

template<int CONSUMERS, int B_LANES, int QSIZE>
void __device__ _tsr_consumer_fused(
    fp8* A_buffer,
    fp8* B_buffer,
    half* D,
    float scale,
    uint8* queue,
    int &index,
    uint8 &p_state,
    int &role_id,
    const int n,
    const int dropped_rows,
    const int dropped_cols,
    const int k,
    const int k_blocks,
    half* partial_results,  // Buffer for partial results
    int* ready_flags        // Flags to signal when partial results are ready
) {
    // Compute thread position
    const int thread_id = threadIdx.x % WARPSIZE;
    A_buffer += (thread_id / 2) * E_P_BANK + (threadIdx.x % 2) * 32*E_P_BANK * 2;
    B_buffer += (thread_id % 32) * 4 + (thread_id / 32) * 32*E_P_BANK * 4;
    const int sparsity_indices = (threadIdx.x % 2) ? 0x0000EEEE : 0x00004444;

    // Declare input registers
    fp8x8 reg_A[OPS];
    fp8x16 reg_B[OPS];

    // Initialize output registers
    f32x4 reg_D[B_LANES];
    #pragma unroll
    for (int i = 0; i < (B_LANES); i++) {
        reg_D[i][0] = 0.0f; reg_D[i][1] = 0.0f; reg_D[i][2] = 0.0f; reg_D[i][3] = 0.0f;
    }

    // K-wise loop
    fp8 *A_offs_buff, *B_offs_buff;
    int b = role_id;

    while (b < k_blocks) {

        // Account for cyclic queue
        index -= (index >= QSIZE) ? QSIZE : 0;
        A_offs_buff = A_buffer + index * (WARPTILE_M * WARPTILE_K);
        B_offs_buff = B_buffer + index * ((OP_N * B_LANES) * WARPTILE_K);

        // Wait for A buffer to be filled
        while (queue[2 * B_LANES * index] != p_state) {
            asm volatile("s_sleep 0");
        }
        // Load A buffer
        #pragma unroll
        for (int op = 0; op < OPS; op++) {
            consumer_smem_to_reg8(A_offs_buff + (op * OP_M * OP_K), reg_A[op]);
        }
        // Mark A buffer as consumed
        queue[2 * B_LANES * index] = p_state + 32;

        // Go through each lanes
        #pragma unroll
        for (int lane = 0; lane < B_LANES; lane++) {

            // Wait for B buffer to be filled
            while (queue[2 * (B_LANES * index + lane) + 1] != p_state) {
                asm volatile("s_sleep 0");
            }
            // Load B buffer
            #pragma unroll
            for (int op = 0; op < OPS; op++) {
                consumer_smem_to_reg16(B_offs_buff + (lane * OP_N * WARPTILE_K) + (op * OP_N * OP_K), reg_B[op]);
            }
            // Mark B buffer as consumed
            queue[2 * (B_LANES * index + lane) + 1] = p_state + 32;

            // Consume registers
            #pragma unroll
            for (int op = 0; op < OPS; op++) {
                for (int i = 0; i < 16; i++) {
                    for (int j = 0; j < 16; j++) {
                        float sum = 0.0f;
                        #pragma unroll
                        for (int k = 0; k < 16; k++) {
                            // Convert fp8 to float and perform multiplication
                            sum += fp8_to_float(reg_A[op][i * 16 + k]) * fp8_to_float(reg_B[op][k * 16 + j]);
                        }
                        // Accumulate the result
                        reg_D[lane][i * 16 + j] += sum;
                    }
                }
            }
        }

        // Update index
        index += CONSUMERS;
        p_state = (index >= QSIZE) ? p_state + 64 : p_state;
        b += CONSUMERS;
    }

    // Bring warps back in order
    role_id = b - k_blocks;

    // Infer the current column in D
    int out_n = 2 * ((thread_id % 16) / 2);

    // Finalize registers so that each threads hold 2 consecutive results in memory
    float final_scale;
    int id_to_swap = 1 - threadIdx.x % 2;
    int src_lane = thread_id + 1 - 2 * (thread_id % 2);

    #pragma unroll
    for (int i = 0; i < B_LANES; i++) {

        // Fusing
        reg_D[i][0] = reg_D[i][0] + reg_D[i][1];
        reg_D[i][1] = reg_D[i][2] + reg_D[i][3];

        // Scaling
        final_scale = (out_n + i * OP_N) >= dropped_cols ? scale : 0.0f;
        reg_D[i][0] *= final_scale;
        reg_D[i][1] *= final_scale;

        // Swapping 
        reg_D[i][id_to_swap] = __shfl(reg_D[i][id_to_swap], src_lane);
    }

    // Infer the current row in D
    int out_m = (thread_id / 16) * 2 + (thread_id % 2);

    // If we are in dropped rows territory, we can return now
    if (out_m + dropped_rows > WARPTILE_M -1) {
        return ;
    }

    // Relocate on D
    __half2* D_ = reinterpret_cast<__half2*>(D) + (out_m * n + out_n) / 2;

    // Out lane by lane
    __half2 x;

    #pragma unroll
    for (int i = 0; i < B_LANES; i++) {
        // Form the packed f16
        x.x = __float2half(reg_D[i][0]);
        x.y = __float2half(reg_D[i][1]);

        // Convert __half2 to float2
        float2 x_float2;
        x_float2.x = __half2float(x.x);
        x_float2.y = __half2float(x.y);

        // Perform atomic addition in FP32
        atomicAdd(reinterpret_cast<float*>(&D_[i * OP_N / 2]), x_float2.x);
        atomicAdd(reinterpret_cast<float*>(&D_[i * OP_N / 2]) + 1, x_float2.y);
    }

    // Store partial results and signal ready
    if (threadIdx.x < CONSUMERS * WARPSIZE) {
        int consumer_id = threadIdx.x / WARPSIZE;
        int lane_id = threadIdx.x % WARPSIZE;
        
        // Add curr_n calculation here
        const int curr_n = blockIdx.y * OP_N * B_LANES;
        
        int offset = consumer_id * B_LANES * 8 * OP_N * B_LANES + lane_id * 8 * OP_N * B_LANES;
        for (int i = 0; i < B_LANES; i++) {
            for (int j = 0; j < 8; j++) {
                for (int k = 0; k < OP_N * B_LANES; k++) {
                    partial_results[offset + i * 8 * OP_N * B_LANES + j * OP_N * B_LANES + k] 
                        = D[curr_n + i * OP_N * B_LANES + k];
                }
            }
        }

        // Signal that partial results are ready
        if (lane_id == 0) {
            atomicAdd(&ready_flags[consumer_id], 1);  // Use atomic operation to set flag
        }
    }
}
