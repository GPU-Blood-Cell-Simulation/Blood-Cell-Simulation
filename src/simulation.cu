#include "simulation.cuh"
#include "grid.cuh"
#include "defines.cuh"

#include <ctime>

namespace sim
{

    __global__ void setupCurandStatesKernel(curandState* states, const unsigned long seed, const int particleCount);

    __global__ void generateRandomPositionsKernel(curandState* states, float* positionX, float* positionY, float* positionZ, const int particleCount);


    // Allocate GPU buffers for the position vectors
    void allocateMemory(float** positionX, float** positionY, float** positionZ,
        unsigned int** cellIds, unsigned int** particleIds, unsigned int** cellStarts, unsigned int** cellEnds,
        const unsigned int particleCount)
    {
        HANDLE_ERROR(cudaMalloc((void**)positionX, particleCount * sizeof(float)));

        HANDLE_ERROR(cudaMalloc((void**)positionY, particleCount * sizeof(float)));

        HANDLE_ERROR(cudaMalloc((void**)positionZ, particleCount * sizeof(float)));

        HANDLE_ERROR(cudaMalloc((void**)cellIds, particleCount * sizeof(unsigned int)));
        HANDLE_ERROR(cudaMalloc((void**)particleIds, particleCount * sizeof(unsigned int)));

        HANDLE_ERROR(cudaMalloc((void**)cellStarts, width / cellWidth * height / cellHeight * depth / cellDepth * sizeof(unsigned int)));
        HANDLE_ERROR(cudaMalloc((void**)cellEnds, width / cellWidth * height / cellHeight * depth / cellDepth * sizeof(unsigned int)));
    }

    // Generate initial positions and velocities of particles
    void generateRandomPositions(float* positionX, float* positionY, float* positionZ, const int particleCount)
    {
        int threadsPerBlock = particleCount > 1024 ? 1024 : particleCount;
        int blocks = (particleCount + threadsPerBlock - 1) / threadsPerBlock;

        // Set up random seeds
        curandState* devStates;
        cudaMalloc(&devStates, particleCount * sizeof(curandState));
        srand(static_cast<unsigned int>(time(0)));
        int seed = rand();
        setupCurandStatesKernel <<<blocks, threadsPerBlock >>>(devStates, seed, particleCount);

        // Generate random positions and velocity vectors

        generateRandomPositionsKernel <<<blocks, threadsPerBlock >>>(devStates, positionX, positionY, positionZ, particleCount);

        cudaFree(devStates);
    }

    __global__ void setupCurandStatesKernel(curandState* states, const unsigned long seed, const int particleCount)
    {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id >= particleCount)
            return;
        curand_init(seed, id, 0, &states[id]);
    }

    // Generate random positions and velocities at the beginning
    __global__ void generateRandomPositionsKernel(curandState* states, float* positionX, float* positionY, float* positionZ, const int particleCount)
    {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id >= particleCount)
            return;

        positionX[id] = curand_uniform(&states[id]) * width;
        positionY[id] = curand_uniform(&states[id]) * height;
        positionZ[id] = curand_uniform(&states[id]) * depth;
    }

    void calculateNextFrame(float* positionX, float* positionY, float* positionZ,
        unsigned int* cellIds, unsigned int* particleIds,
        unsigned int* cellStarts, unsigned int* cellEnds, unsigned int particleCount)
    {
        // 1. calculate grid
        createUniformGrid(positionX, positionY, positionZ, cellIds, particleIds, cellStarts, cellEnds, particleCount);

        // 2. TODO: detections
    }
}
