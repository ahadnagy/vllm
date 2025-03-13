#include <torch/extension.h>
#include <vector>
#include <mpi.h>
#include <mscclpp/core.hpp>
#include <mscclpp/utils.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/memory_channel.hpp>
#include <c10/hip/HIPStream.h>

#include "skinny_gemm/skinny_gemm.cu"

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

    torch::Tensor reduce(
        torch::Tensor& A,
        torch::Tensor& B,
        torch::Tensor& D,
        torch::Tensor& scale_tensor,
        int64_t b_lanes,
        int64_t split_k) {
        TORCH_CHECK(A.is_cuda(), "Input tensor must be a CUDA tensor");
        TORCH_CHECK(A.is_contiguous(), "Input tensor must be contiguous");
 
        // Setup mesh connections
        allocateCommsBuffers(D.numel() * D.element_size());
        printf("Allocated input buffers\n");
        setupMeshConnections();
        printf("Setup mesh connections\n");

        CUDATHROW(cudaDeviceSynchronize());
        skinny_gemm(A, B, D, scale_tensor, b_lanes, split_k, rank_, worldSize_, recv_buff_.get(), send_buff_.get());
        CUDATHROW(cudaDeviceSynchronize());
        return D;
    }

private:
    void bootstrap() {
        // Use longer timeout for initialization

        //MPI_Init(NULL, NULL);
        //MPI_Comm_size(MPI_COMM_WORLD, &worldSize_);
        //MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

        printf("Rank %d: World size %d\n", rank_, worldSize_);

        std::string ip_port = "localhost:12000";
        auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank_, worldSize_);
        
        // Initialize with options
        //bootstrap->initialize(ip_port, options);
        //mscclpp::UniqueId id;
        //if (bootstrap->getRank() == 0) id = bootstrap->createUniqueId();
        //MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        //bootstrap->initialize(id);
        bootstrap->initialize("127.0.0.1:50000");
        bootstrap->barrier();
        printf("Initialized comms\n");
        
        // Create communicator and wait for all processes
        communicator_ = std::make_shared<mscclpp::Communicator>(bootstrap);
        chanService_ = std::make_shared<mscclpp::ProxyService>();
    }

    void allocateCommsBuffers(size_t bytes) {
        recv_buff_ = mscclpp::GpuBuffer<uint8_t>(bytes).memory();
        send_buff_ = mscclpp::GpuBuffer<uint8_t>(bytes).memory();
        comms_buff_bytes_ = bytes;
    }

    void setupMeshConnections() {
        mscclpp::Transport transport = mscclpp::Transport::CudaIpc;
        std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteRegMemories;
        std::vector<mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>> connectionFutures;

        printf("Rank %d: Setting up mesh connections\n", rank_);
        mscclpp::RegisteredMemory recvBufRegMem = communicator_->registerMemory(recv_buff_.get(), comms_buff_bytes_, transport);
        mscclpp::RegisteredMemory sendBufRegMem = communicator_->registerMemory(send_buff_.get(), comms_buff_bytes_, transport);
        printf("Registered memory\n");

        // Connect with all other ranks
        for (int r = 0; r < worldSize_; ++r) {
            if (r == rank_) continue;
            connectionFutures.push_back(communicator_->connectOnSetup(r, 0, transport));
            communicator_->sendMemoryOnSetup(recvBufRegMem, r, 0);
            remoteRegMemories.push_back(communicator_->recvMemoryOnSetup(r, 0));
        }
        printf("Connected with all other ranks\n");

        communicator_->setup();

        printf("Setup communicator\n");

        std::transform(
            connectionFutures.begin(), connectionFutures.end(), std::back_inserter(connections_),
            [](const mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>& future) { return future.get(); });
        printf("Got connections\n");


        auto service = std::dynamic_pointer_cast<mscclpp::ProxyService>(chanService_);
        for (size_t i = 0; i < connections_.size(); ++i) {
            channels_.push_back(mscclpp::deviceHandle(
                service->portChannel(service->buildAndAddSemaphore(*communicator_, connections_[i]),
                                     service->addMemory(remoteRegMemories[i].get()), service->addMemory(sendBufRegMem))));
        }

        printf("Created channels: %d\n", channels_.size());

        communicator_->setup();

        CUDATHROW(cudaMemcpyToSymbol(constRingChannels, channels_.data(),
                                 sizeof(DeviceHandle<mscclpp::PortChannel>) * channels_.size()));

        printf("Copied channels to device\n");

        this->chanService_->startProxy();
        communicator_->bootstrap()->barrier();
        printf("Started proxy\n");
    }


    int rank_;
    int worldSize_;
    std::shared_ptr<mscclpp::Communicator> communicator_;
    std::vector<DeviceHandle<mscclpp::PortChannel>> channels_;
    std::vector<std::shared_ptr<mscclpp::Connection>> connections_;
    std::shared_ptr<mscclpp::BaseProxyService> chanService_;
    cudaStream_t stream_;

    std::shared_ptr<uint8_t> recv_buff_;
    std::shared_ptr<uint8_t> send_buff_;
    size_t comms_buff_bytes_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<AllReduceEngine>(m, "AllReduceEngine")
        .def(py::init<int, int>())
        .def("reduce", &AllReduceEngine::reduce);
} 