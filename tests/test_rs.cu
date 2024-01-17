// nvcc -O2 -I /usr/local/mpi/include -I ~/nccl/build/include -L /usr/local/mpi/lib -L ~/nccl/build/lib -gencode=arch=compute_80,code=sm_80 -o test_rs test_rs.cu -lnccl -lmpi
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cmath>
#include <cstdlib>
#include <cstdio>

#include <nccl.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <mpi.h>

void NcclOrDie(ncclResult_t result, const char* action) {
    if (result != ncclSuccess) {
        std::cerr << action << ": " << ncclGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CudaOrDie(cudaError_t result, const char* action) {
    if (result != cudaSuccess) {
        std::cerr << action << ": " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

enum class ReduceScatterMode {
    PureBfloat16 = 0,
    Float32Acc,
    Float32Upcast,
    ModeCount,
};

template <typename Src, typename Dst>
__global__ void Convert(const Src* src, Dst* dst, size_t count) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = src[idx];
    }
}

constexpr size_t WARMUP_ITERS = 10;
constexpr size_t ITERS = 100;

std::vector<__nv_bfloat16> DoReduceScatter(__nv_bfloat16* sendbuff, size_t elemCount, size_t recvCount, ncclComm_t comm, cudaStream_t stream,
                                           ReduceScatterMode mode, int rank, float* elapsedMs) {
    __nv_bfloat16* recvbuff = sendbuff + rank * recvCount;
    float* tmpbuff = {};

    constexpr size_t N_THREADS = 1024;
    const size_t nBlocks = elemCount / N_THREADS + (elemCount % N_THREADS != 0);

    cudaEvent_t startEvent{}, endEvent{};
    CudaOrDie(cudaEventCreate(&startEvent), "creating start event");
    CudaOrDie(cudaEventCreate(&endEvent), "creating end event");

    if (mode == ReduceScatterMode::PureBfloat16) {
        CudaOrDie(cudaEventRecord(startEvent, stream), "record start event");
        NcclOrDie(
            ncclReduceScatter(sendbuff, recvbuff, recvCount, ncclBfloat16, ncclSum, comm, stream),
            "reduce-scatter"
        );
        CudaOrDie(cudaEventRecord(endEvent, stream), "record end event");
    } else if (mode == ReduceScatterMode::Float32Acc) {
        CudaOrDie(cudaEventRecord(startEvent, stream), "record start event");
        NcclOrDie(
            ncclMixedPrecisionReduceScatter(sendbuff, recvbuff, recvCount, ncclFloat, ncclBfloat16, ncclSum, comm, stream),
            "reduce-scatter"
        );
        CudaOrDie(cudaEventRecord(endEvent, stream), "record end event");
    } else {
        if (mode != ReduceScatterMode::Float32Upcast) {
            std::cerr << "Invalid mode: " << static_cast<unsigned long long>(mode) << std::endl;
            exit(EXIT_FAILURE);
        }

        CudaOrDie(cudaMalloc(&tmpbuff, elemCount * sizeof(float)), "allocate temporary FP32 buffer");
        Convert<__nv_bfloat16, float><<<nBlocks, N_THREADS, 0, stream>>>(sendbuff, tmpbuff, elemCount);
        CudaOrDie(cudaGetLastError(), "bf16 -> fp32 conversion");

        CudaOrDie(cudaEventRecord(startEvent, stream), "record start event");
        NcclOrDie(
            ncclReduceScatter(tmpbuff, tmpbuff + rank * recvCount, recvCount, ncclFloat, ncclSum, comm, stream),
            "reduce-scatter"
        );
        CudaOrDie(cudaEventRecord(endEvent, stream), "record end event");

        Convert<float, __nv_bfloat16><<<nBlocks, N_THREADS, 0, stream>>>(tmpbuff, sendbuff, elemCount);
        CudaOrDie(cudaGetLastError(), "fp32 -> bf16 conversion");
    }

    CudaOrDie(cudaStreamSynchronize(stream), "synchronize GPU");

    std::vector<__nv_bfloat16> result(recvCount);
    CudaOrDie(
        cudaMemcpy(result.data(), recvbuff, recvCount * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost),
        "copy the final result from GPU"
    );
    CudaOrDie(cudaFree(tmpbuff), "freeing temporary FP32 buffers, if any");
    CudaOrDie(cudaEventElapsedTime(elapsedMs, startEvent, endEvent), "get elapsed time");
    return result;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <number of elements>" << std::endl;
        return EXIT_FAILURE;
    }

    const auto elemCount = std::stoull(argv[1]);

    int nranks{}, rank{}, localRank{};
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const auto* localRankEnv = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (localRankEnv == nullptr) {
        std::cerr << "OMPI_COMM_WORLD_LOCAL_RANK is not set" << std::endl;
        return EXIT_FAILURE;
    }
    localRank = std::stoi(localRankEnv);

    if (elemCount % nranks != 0) {
        std::cerr << "The number of elements (" << elemCount << ") is not divisible by the number of ranks (" << nranks << ")" << std::endl;
        return EXIT_FAILURE;
    }
    size_t recvCount = elemCount / nranks;

    ncclUniqueId ncclId{};
    if (rank == 0) {
        NcclOrDie(ncclGetUniqueId(&ncclId), "get NCCL ID");
    }
    MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<__nv_bfloat16> gradients(elemCount);
    std::default_random_engine generator(31337 + rank);
    std::normal_distribution<double> distribution(0.0, 0.02);

    for (auto& x : gradients) {
        x = distribution(generator);
    }

    std::vector<float> fp32Gradients(gradients.begin(), gradients.end());
    // Use the AllReduce from MPI as the reference result
    MPI_Allreduce(MPI_IN_PLACE, fp32Gradients.data(), elemCount, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    ncclComm_t comm{};
    CudaOrDie(cudaSetDevice(localRank), "set current GPU");
    NcclOrDie(ncclCommInitRank(&comm, nranks, ncclId, rank), "initialize NCCL communicator");

    __nv_bfloat16* sendbuff;
    cudaStream_t stream;
    CudaOrDie(cudaStreamCreate(&stream), "create GPU stream");
    CudaOrDie(cudaMalloc(&sendbuff, elemCount * sizeof(__nv_bfloat16)), "allocate GPU buffer");

    for (size_t rawMode = 0; rawMode < static_cast<size_t>(ReduceScatterMode::ModeCount); ++rawMode) {
        const auto mode = static_cast<ReduceScatterMode>(rawMode);

        float totalMaxDiff = 0.0f;
        float totalElapsedMs = 0.0f;
        for (size_t iter = 0; iter < ITERS; ++iter) {
            float elapsedMs{};

            CudaOrDie(
                cudaMemcpy(sendbuff, gradients.data(), elemCount * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice),
                "copy to GPU"
            );
            const auto result = DoReduceScatter(sendbuff, elemCount, recvCount, comm, stream, mode, rank, &elapsedMs);

            if (iter < WARMUP_ITERS) {
                continue;
            }

            float maxDiff = 0.0f;
            for (size_t off = 0; off < recvCount; ++off) {
                const float expected = fp32Gradients[rank*recvCount + off];
                const float actual = result[off];

                const auto diff = std::abs(expected - actual);

                if (std::isnan(diff)) {
                    std::cerr << "NaN detected at index " << off << std::endl;
                    return EXIT_FAILURE;
                }

                if (diff > maxDiff) {
                    maxDiff = diff;
                }
            }

            MPI_Allreduce(MPI_IN_PLACE, &maxDiff, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &elapsedMs, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

            if (maxDiff > totalMaxDiff) {
                totalMaxDiff = maxDiff;
            }
            // The maximum kernel time for the collective is the total time ReduceScatter takes
            totalElapsedMs += elapsedMs;
        }

        totalElapsedMs /= (ITERS - WARMUP_ITERS);

        if (rank == 0) {
            std::cout << "Mode = " << rawMode << ", max diff = " << totalMaxDiff << ", avg max kernel time (ms) = " << totalElapsedMs << std::endl;
        }
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
