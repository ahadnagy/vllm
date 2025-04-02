#include <torch/extension.h>
#include <vector>
#include <string>
//#include <mpi.h>
#include <mscclpp/core.hpp>
#include <mscclpp/utils.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/memory_channel.hpp>
#include <c10/hip/HIPStream.h>

#include "skinny_gemm/skinny_gemm.cu"

class AllReduceEngine {
public:
    AllReduceEngine(int rank, int worldSize,
                    int port,
                    torch::Tensor& comms_buff_A,
                    torch::Tensor& comms_buff_B)
        : rank_(rank), worldSize_(worldSize), comms_buff_A_(comms_buff_A.data_ptr()), comms_buff_B_(comms_buff_B.data_ptr()) {
        bootstrap(port);
        printf("Allocated input buffers\n");
        comms_buff_bytes_ = comms_buff_A.numel() * comms_buff_A.element_size();
        setupMeshConnections(channels_A_, comms_buff_A.data_ptr(), comms_buff_B.data_ptr(), comms_buff_bytes_);
        CUDATHROW(cudaMemcpyToSymbol(constRingChannelsA, channels_A_.data(),
            sizeof(DeviceHandle<mscclpp::PortChannel>) * channels_A_.size()));
        printf("Copied channels to device\n");
        setupMeshConnections(channels_B_, comms_buff_B.data_ptr(), comms_buff_A.data_ptr(), comms_buff_bytes_);
        CUDATHROW(cudaMemcpyToSymbol(constRingChannelsB, channels_B_.data(),
            sizeof(DeviceHandle<mscclpp::PortChannel>) * channels_B_.size()));
        printf("Copied channels to device\n");

        printf("Setup mesh connections\n");
        startProxy();

        //cudaEventCreate(&allreduce_lock_event);
        //cudaEventRecord(allreduce_lock_event, at::cuda::getCurrentCUDAStream());
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
        int64_t split_k,
        bool is_capturing) {
        TORCH_CHECK(A.is_cuda(), "Input tensor must be a CUDA tensor");
        TORCH_CHECK(A.is_contiguous(), "Input tensor must be contiguous");

        // Setup mesh connections
        //allocateCommsBuffers(D.numel() * D.element_size());
        // printf("Allocated input buffers\n");
        // comms_buff_bytes_ = D.numel() * D.element_size();
        // setupMeshConnections(channels_A_, comms_buff_A.data_ptr(), comms_buff_B.data_ptr(), comms_buff_bytes_);
        // CUDATHROW(cudaMemcpyToSymbol(constRingChannelsA, channels_A_.data(),
        //     sizeof(DeviceHandle<mscclpp::PortChannel>) * channels_A_.size()));
        // printf("Copied channels to device\n");
        // setupMeshConnections(channels_B_, comms_buff_B.data_ptr(), comms_buff_A.data_ptr(), comms_buff_bytes_);
        // CUDATHROW(cudaMemcpyToSymbol(constRingChannelsB, channels_B_.data(),
        //     sizeof(DeviceHandle<mscclpp::PortChannel>) * channels_B_.size()));
        // printf("Copied channels to device\n");

        // printf("Setup mesh connections\n");
        // startProxy();

        CUDATHROW(hipMemsetD16(reinterpret_cast<uint8_t *>(comms_buff_A_), 0, A.numel()));
        CUDATHROW(hipMemsetD16(reinterpret_cast<uint8_t *>(comms_buff_B_), 0, A.numel()));
        CUDATHROW(cudaDeviceSynchronize());
        communicator_->bootstrap()->barrier();

        skinny_gemm(A, B, D, scale_tensor, b_lanes, split_k, rank_, worldSize_, reinterpret_cast<uint8_t *>(comms_buff_A_), reinterpret_cast<uint8_t *>(comms_buff_B_), allreduce_lock_event, is_capturing);
        CUDATHROW(cudaDeviceSynchronize());
        communicator_->bootstrap()->barrier();
        return D;
    }

private:
    void bootstrap(int port) {
        // Use longer timeout for initialization

        //MPI_Init(NULL, NULL);
        //MPI_Comm_size(MPI_COMM_WORLD, &worldSize_);
        //MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

        printf("Rank %d: World size %d\n", rank_, worldSize_);

        std::string ip_port = "localhost:";
        auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank_, worldSize_);

        // Initialize with options
        //bootstrap->initialize(ip_port, options);
        //mscclpp::UniqueId id;
        //if (bootstrap->getRank() == 0) id = bootstrap->createUniqueId();
        //MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        //bootstrap->initialize(id);
        bootstrap->initialize(ip_port.append(std::to_string(port)));
        bootstrap->barrier();
        printf("Initialized comms\n");

        // Create communicator and wait for all processes
        communicator_ = std::make_shared<mscclpp::Communicator>(bootstrap);
        chanService_ = std::make_shared<mscclpp::ProxyService>();
    }

    void allocateCommsBuffers(size_t bytes) {
        comm_buff_A = mscclpp::GpuBuffer<uint8_t>(bytes).memory();
        comm_buff_B = mscclpp::GpuBuffer<uint8_t>(bytes).memory();
        comms_buff_bytes_ = bytes;
    }

    void setupMeshConnections(std::vector<DeviceHandle<mscclpp::PortChannel>>& portChannels, void* send_buff, void* recv_buff, size_t buff_size) {
        mscclpp::Transport transport = mscclpp::Transport::CudaIpc;
        std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteRegMemories;
        std::vector<mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>> connectionFutures;
        std::vector<std::shared_ptr<mscclpp::Connection>> connections;

        printf("Rank %d: Setting up mesh connections\n", rank_);
        mscclpp::RegisteredMemory recvBufRegMem = communicator_->registerMemory(recv_buff, buff_size, transport);
        mscclpp::RegisteredMemory sendBufRegMem = communicator_->registerMemory(send_buff, buff_size, transport);
        //mscclpp::RegisteredMemory bufRegMem = communicator_->registerMemory(buff, buff_size, transport);
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
            connectionFutures.begin(), connectionFutures.end(), std::back_inserter(connections),
            [](const mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>& future) { return future.get(); });
        printf("Got connections\n");


        auto service = std::dynamic_pointer_cast<mscclpp::ProxyService>(chanService_);
        for (size_t i = 0; i < connections.size(); ++i) {
            portChannels.push_back(mscclpp::deviceHandle(
                service->portChannel(service->buildAndAddSemaphore(*communicator_, connections[i]),
                                     service->addMemory(remoteRegMemories[i].get()), service->addMemory(sendBufRegMem))));
        }

        printf("Created channels: %d\n", portChannels.size());

        communicator_->setup();
    }

    void startProxy() {
        this->chanService_->startProxy();
        communicator_->bootstrap()->barrier();
        printf("Started proxy\n");
    }


    int rank_;
    int worldSize_;
    std::shared_ptr<mscclpp::Communicator> communicator_;
    std::vector<DeviceHandle<mscclpp::PortChannel>> channels_A_;
    std::vector<DeviceHandle<mscclpp::PortChannel>> channels_B_;
    std::shared_ptr<mscclpp::BaseProxyService> chanService_;
    cudaStream_t stream_;

    std::shared_ptr<uint8_t> comm_buff_A;
    std::shared_ptr<uint8_t> comm_buff_B;

    void* comms_buff_A_;
    void* comms_buff_B_;
    size_t comms_buff_bytes_;

    cudaEvent_t allreduce_lock_event;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<AllReduceEngine>(m, "AllReduceEngine")
        .def(py::init<int, int, int, torch::Tensor&, torch::Tensor&>())
        .def("reduce", &AllReduceEngine::reduce);
}
