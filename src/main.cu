
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstdlib>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <curand.h>
#include <curand_kernel.h>


constexpr float width = 100.0f;
constexpr float height = 100.0f;
constexpr float depth = 100.0f;
constexpr int cellWidth = 2;
constexpr int cellHeight = 2;
constexpr int cellDepth = 2;

// A handy function for easy error checking
static void HandleError(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
            file, line);
        cudaDeviceSynchronize();
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ));

void allocateMemory(float** positionX, float** positionY, float** positionZ, const unsigned int particleCount);
void createUniformGrid(const float* _positionX, const float* _positionY, const float* _positionZ, const unsigned int particleCount);
void generateRandomPositions(float* positionX, float* positionY, float* positionZ, const int particleCount);

__global__ void setupCurandStatesKernel(curandState* states, const unsigned long seed, const int particleCount);

__global__ void generateRandomPositionsKernel(curandState* states, float* positionX, float* positionY, float* positionZ, const int particleCount);

__global__ void calculateCellIdKernel(const float* positionX, const float* positionY, const float* positionZ,
    unsigned int* CellIds, unsigned int* ParticleIds, const unsigned int particleCount);

__global__ void calculateStartAndEndOfCellKernel(const float* positionX, const float* positionY, const float* positionZ,
    const unsigned int* cellIds, const unsigned int* particleIds,
    unsigned int* cellStarts, unsigned int* cellEnds, unsigned int particleCount);
int main()
{
    constexpr unsigned int particleCount = 500;

    float* positionX = 0;
    float* positionY = 0;
    float* positionZ = 0;

    // Choose which GPU to run on, change this on a multi-GPU system.
    HANDLE_ERROR(cudaSetDevice(0));

    // Allocate memory
    allocateMemory(&positionX, &positionY, &positionZ, particleCount);

    generateRandomPositions(positionX, positionY, positionZ, particleCount);

    // Add vectors in parallel.
    createUniformGrid(positionX, positionY, positionZ, particleCount);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.

    cudaFree(positionX);
    cudaFree(positionY);
    cudaFree(positionZ);

    HANDLE_ERROR(cudaDeviceReset());

    return 0;
}

// Allocate GPU buffers for the position vectors
void allocateMemory(float** positionX, float** positionY, float** positionZ, const unsigned int particleCount)
{
    HANDLE_ERROR(cudaMalloc((void**)positionX, particleCount * sizeof(float)));

    HANDLE_ERROR(cudaMalloc((void**)positionY, particleCount * sizeof(float)));

    HANDLE_ERROR(cudaMalloc((void**)positionZ, particleCount * sizeof(float)));
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
    setupCurandStatesKernel << <blocks, threadsPerBlock >> > (devStates, seed, particleCount);

    // Generate random positions and velocity vectors

    generateRandomPositionsKernel << <blocks, threadsPerBlock >> > (devStates, positionX, positionY, positionZ, particleCount);

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

    positionX[id] = curand_uniform(&states[id]) * 100;
    positionY[id] = curand_uniform(&states[id]) * 100;
    positionZ[id] = curand_uniform(&states[id]) * 100;
}

void createUniformGrid(const float* positionX, const float* positionY, const float* positionZ, const unsigned int particleCount)
{
    // Calculate launch parameters

    const int threadsPerBlock = particleCount > 1024 ? 1024 : particleCount;
    const int blocks = (particleCount + threadsPerBlock - 1) / threadsPerBlock;

    // 0. Allocate buffers for key-value pairs and cell start/end buffers

    unsigned int* cellIds = 0;
    unsigned int* particleIds = 0;

    HANDLE_ERROR(cudaMalloc((void**)&cellIds, particleCount * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void**)&particleIds, particleCount * sizeof(unsigned int)));

    unsigned int* cellStarts = 0;
    unsigned int* cellEnds = 0;

    HANDLE_ERROR(cudaMalloc((void**)&cellStarts, width / cellWidth * height / cellHeight * depth / cellDepth * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void**)&cellEnds, width / cellWidth * height / cellHeight * depth / cellDepth * sizeof(unsigned int)));

    // 1. Calculate cell id for every particle and store as pair (cell id, particle id) in two buffers
    calculateCellIdKernel << <blocks, threadsPerBlock >> >
        (positionX, positionY, positionZ, cellIds, particleIds, particleCount);

    // 2. Sort particle ids by cell id

    thrust::device_ptr<unsigned int> keys = thrust::device_pointer_cast<unsigned int>(cellIds);
    thrust::device_ptr<unsigned int> values = thrust::device_pointer_cast<unsigned int>(particleIds);

    thrust::stable_sort_by_key(keys, keys + particleCount, values);
    thrust::sort(keys, keys + particleCount);

    // 3. Find the start and end of every cell

    calculateStartAndEndOfCellKernel << <blocks, threadsPerBlock >> >
        (positionX, positionY, positionZ, cellIds, particleIds, cellStarts, cellEnds, particleCount);

    cudaFree(cellIds);
    cudaFree(particleIds);
    cudaFree(cellStarts);
    cudaFree(cellEnds);
}

__global__ void calculateCellIdKernel(const float* positionX, const float* positionY, const float* positionZ,
    unsigned int* cellIds, unsigned int* particleIds, const unsigned int particleCount)
{
    unsigned int particleId = blockIdx.x * blockDim.x + threadIdx.x;
    if (particleId >= particleCount)
        return;

    unsigned int cellId =
        static_cast<unsigned int>(positionX[particleId] / cellWidth) +
        static_cast<unsigned int>(positionY[particleId] / cellHeight) +
        static_cast<unsigned int>(positionZ[particleId] / cellDepth);

    particleIds[particleId] = particleId;
    cellIds[particleId] = cellId;

}

__global__ void calculateStartAndEndOfCellKernel(const float* positionX, const float* positionY, const float* positionZ,
    const unsigned int* cellIds, const unsigned int* particleIds,
    unsigned int* cellStarts, unsigned int* cellEnds, unsigned int particleCount)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= particleCount)
        return;

    unsigned int currentCellId = cellIds[id];

    // Check if the previous cell id was different - it would mean we found the start of a cell
    if (id > 0 && currentCellId != cellIds[id - 1])
    {
        cellStarts[currentCellId] = id;
    }

    // Check if the next cell id was different - it would mean we found the end of a cell
    if (id < particleCount - 1 && currentCellId != cellIds[id + 1])
    {
        cellEnds[currentCellId] = id;
    }

    if (id < 5) printf("%d. particle id: %d, cell: %d\n", id, particleIds[id], cellIds[id]);

}
