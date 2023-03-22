
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

void allocateMemory(float** devPositionX, float** devPositionY, float** devPositionZ, const unsigned int particleCount);
void createUniformGrid(const float* dev_positionX, const float* dev_positionY, const float* dev_positionZ, const unsigned int particleCount);

__global__ void calculateCellIdKernel(const float* devPositionX, const float* devPositionY, const float* devPositionZ, const unsigned int particleCount);

__global__ void calculateBeginningAndEndOfCellKernel(const float* devPositionX, const float* devPositionY, const float* devPositionZ, const unsigned int particleCount);

int main()
{
    constexpr unsigned int particleCount = 10;

    float* devPositionX = 0;
    float* devPositionY = 0;
    float* devPositionZ = 0;

    // Choose which GPU to run on, change this on a multi-GPU system.
    HANDLE_ERROR(cudaSetDevice(0));

    // Allocate memory
    allocateMemory(&devPositionX, &devPositionY, &devPositionZ, particleCount);

    // Add vectors in parallel.
    createUniformGrid(devPositionX, devPositionY, devPositionZ, particleCount);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.

    cudaFree(devPositionX);
    cudaFree(devPositionY);
    cudaFree(devPositionZ);

    HANDLE_ERROR(cudaDeviceReset());

    return 0;
}

// Allocate GPU buffers for the position vectors
void allocateMemory(float** devPositionX, float** devPositionY, float** devPositionZ, const unsigned int particleCount)
{
    HANDLE_ERROR(cudaMalloc((void**)devPositionX, particleCount * sizeof(float)));

    HANDLE_ERROR(cudaMalloc((void**)devPositionY, particleCount * sizeof(float)));

    HANDLE_ERROR(cudaMalloc((void**)devPositionZ, particleCount * sizeof(float)));
}

void createUniformGrid(const float* dev_positionX, const float* dev_positionY, const float* dev_positionZ, const unsigned int particleCount)
{
    // Calculate launch parameters

    const int threadsPerBlock = particleCount > 1024 ? 1024 : particleCount;
    const int blocks = (particleCount + threadsPerBlock - 1) / threadsPerBlock;

    // 0. Allocate buffers for key-value pairs

    float* devCellIds = 0;
    float* devParticleIds = 0;

    HANDLE_ERROR(cudaMalloc((void**)&devCellIds, particleCount * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&devParticleIds, particleCount * sizeof(float)));

    // 1. Calculate cell id for every particle and store as pair (cell id, particle id) in two buffers

    // 2. Sort particle ids by cell id

    // 3. Find the start and end of every cell

    cudaFree(devCellIds);
    cudaFree(devParticleIds);
}
