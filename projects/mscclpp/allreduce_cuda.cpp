#include <torch/extension.h>
#include <vector>
#include <mpi.h>
#include <mscclpp/core.hpp>
#include <mscclpp/utils.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/memory_channel.hpp>
#include <c10/hip/HIPStream.h>

// Forward declaration of CUDA functions
// void launch_allreduce(float* data, size_t count, 
//                      mscclpp::DeviceHandle<mscclpp::PortChannel>* channels,
//                      int numChannels, hipStream_t stream);

#define CUDATHROW(cmd)                                                                                                \
  do {                                                                                                                \
    cudaError_t err = cmd;                                                                                            \
    if (err != cudaSuccess) {                                                                                         \
      std::string msg = std::string("Test CUDA failure: ") + std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                        " '" + cudaGetErrorString(err) + "'";                                                         \
      throw std::runtime_error(msg);                                                                                  \
    }                                                                                                                 \
  } while (0)

template <class T>
using DeviceHandle = mscclpp::DeviceHandle<T>;
__device__ __constant__ DeviceHandle<mscclpp::PortChannel> constRingChannels[16];


class AllReduceEngine {
public:
    AllReduceEngine(int rank, int worldSize) 
        : rank_(rank), worldSize_(worldSize) {
        bootstrap();
    }

    ~AllReduceEngine() {
        //if (deviceChannels_) {
        //    //hipFree(deviceChannels_);
        //}
    }

    void reduce(
        torch::Tensor& A,
        torch::Tensor& B,
        torch::Tensor& D,
        torch::Tensor& scale_tensor,
        int64_t b_lanes,
        int64_t split_k) {
        TORCH_CHECK(A.is_cuda(), "Input tensor must be a CUDA tensor");
        TORCH_CHECK(A.is_contiguous(), "Input tensor must be contiguous");


        // Setup mesh connections
        allocateInputBuffers(10);
        setupMeshConnections();

        // Launch allreduce
        
        //hipStream_t stream = at::hip::getCurrentHIPStream(tensor.device().index());
        //launch_allreduce(data_ptr, count, deviceChannels_, 
        //                worldSize_ - 1, stream);
    }

private:
    void bootstrap() {
        // Use longer timeout for initialization

        MPI_Init(NULL, NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &worldSize_);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

        std::string ip_port = "localhost:12000";
        auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank_, worldSize_);
        
        // Initialize with options
        //bootstrap->initialize(ip_port, options);
        mscclpp::UniqueId id;
        if (bootstrap->getRank() == 0) id = bootstrap->createUniqueId();
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        bootstrap->initialize(id);
        
        // Create communicator and wait for all processes
        communicator_ = std::make_shared<mscclpp::Communicator>(bootstrap);
        chanService_ = std::make_shared<mscclpp::ProxyService>();
    }

    void allocateInputBuffers(size_t bytes) {
        input_buff_ = mscclpp::GpuBuffer<int>(bytes / sizeof(int)).memory();
        input_buff_bytes_ = bytes;
    }

    void setupMeshConnections() {
        mscclpp::Transport transport = mscclpp::Transport::CudaIpc;
        std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteRegMemories;
        std::vector<mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>> connectionFutures;

        
        mscclpp::RegisteredMemory inputBufRegMem = communicator_->registerMemory(input_buff_.get(), input_buff_bytes_, transport);


        // Connect with all other ranks
        for (int r = 0; r < worldSize_; ++r) {
            if (r == rank_) continue;
            connectionFutures.push_back(communicator_->connectOnSetup(r, 0, transport));
            communicator_->sendMemoryOnSetup(inputBufRegMem, r, 0);
            remoteRegMemories.push_back(communicator_->recvMemoryOnSetup(r, 0));
        }

        communicator_->setup();

        for (int r = 0; r < worldSize_; ++r) {
            if (r == rank_) continue;
            connections_[r] = connectionFutures[r].get();
        }


        auto service = std::dynamic_pointer_cast<mscclpp::ProxyService>(chanService_);
        for (size_t i = 0; i < connections_.size(); ++i) {
            channels_.push_back(mscclpp::deviceHandle(
                service->portChannel(service->buildAndAddSemaphore(*communicator_, connections_[i]),
                                     service->addMemory(remoteRegMemories[i].get()), service->addMemory(inputBufRegMem))));
        }

        communicator_->setup();

        CUDATHROW(cudaMemcpyToSymbol(constRingChannels, channels_.data(),
                                 sizeof(DeviceHandle<mscclpp::PortChannel>) * channels_.size()));
    }


    // void setupChannels() {
    //     std::string ip_port = "127.0.0.1:50000";
    //     auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank_, worldsize_);
    //     bootstrap->initialize(ip_port);
    //     communicator_ = std::make_shared<mscclpp::Communicator>(bootstrap);
    //     mscclpp::ProxyService proxyService;

    //     communicator_->registerMemory
        
    //     // Create channels for ring communication
    //     for (int i = 0; i < worldSize_; i++) {
    //         if (i == rank_) continue;
            
    //         // Create port channel
    //         auto channel = std::make_unique<mscclpp::MemoryChannel>();
            
    //         // Configure channel
    //         mscclpp::PortConfig config;
    //         config.setRemoteRank(i);
    //         channel->configure(config);
            
    //         // Register memory
    //         channel->registerMemory(*communicator_);
            
    //         channels_.push_back(std::move(channel));
    //     }

    //     // Allocate and copy device handles
    //     hipMalloc(&deviceChannels_, 
    //               channels_.size() * sizeof(mscclpp::DeviceHandle<mscclpp::PortChannel>));
        
    //     // Copy channel handles to device
    //     for (size_t i = 0; i < channels_.size(); i++) {
    //         auto handle = channels_[i]->deviceHandle();
    //         hipMemcpy(&deviceChannels_[i], &handle, 
    //                  sizeof(mscclpp::DeviceHandle<mscclpp::PortChannel>),
    //                  hipMemcpyHostToDevice);
    //     }
    // }


    int rank_;
    int worldSize_;
    std::shared_ptr<mscclpp::Communicator> communicator_;
    std::vector<DeviceHandle<mscclpp::PortChannel>> channels_;
    std::vector<std::shared_ptr<mscclpp::Connection>> connections_;
    std::shared_ptr<mscclpp::BaseProxyService> chanService_;
    cudaStream_t stream_;

    std::shared_ptr<int> input_buff_;
    size_t input_buff_bytes_;
    //mscclpp::DeviceHandle<mscclpp::PortChannel>* deviceChannels_;


    //std::vector<mscclpp::SemaphoreId> semaphoreIds;
    //std::vector<mscclpp::RegisteredMemory> localMemories;
    //std::vector<mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>> connections;
    //std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteMemories;

};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<AllReduceEngine>(m, "AllReduceEngine")
        .def(py::init<int, int>())
        .def("reduce", &AllReduceEngine::reduce);
} 