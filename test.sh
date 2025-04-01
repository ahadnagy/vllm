
# Env
export VLLM_USE_TRITON_FLASH_ATTN=0
export NCCL_MIN_NCHANNELS=112
export VLLM_FP8_PADDING=1

# patch 
cp /home/akos/repos/vllm/vllm/model_executor/models/llama.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/llama.py
#cp /home/akos/repos/vllm/vllm/model_executor/models/llama_unpatch.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/llama.py
cp /home/akos/repos/vllm/vllm/distributed/device_communicators/cuda_communicator.py /usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/cuda_communicator.py
cp /home/akos/repos/vllm/vllm/distributed/parallel_state.py /usr/local/lib/python3.12/dist-packages/vllm/distributed/parallel_state.py
cp /home/akos/repos/vllm/vllm/distributed/__init__.py /usr/local/lib/python3.12/dist-packages/vllm/distributed/__init__.py
cp /home/akos/repos/vllm/vllm/model_executor/model_loader/loader.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/loader.py

# Params
export model="/data/hub/models--amd--Llama-3.1-405B-Instruct-FP8-KV/snapshots/2505537398e7cfda52f6d666f315c03db8e4697c/"

cd /app/vllm/benchmarks
python benchmark_latency.py \
    --batch-size 8 --input-len 1 --output-len 128 --tensor-parallel-size 8 \
    --dtype "float16" --model $model --disable-custom-all-reduce