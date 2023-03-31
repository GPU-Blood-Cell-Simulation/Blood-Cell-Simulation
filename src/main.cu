#include "simulation.cuh"
#include "defines.cuh"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>

int main()
{
    constexpr unsigned int particleCount = 500;

    float* positionX = 0;
    float* positionY = 0;
    float* positionZ = 0;
    unsigned int* cellIds = 0;
    unsigned int* particleIds = 0;
    unsigned int* cellStarts = 0;
    unsigned int* cellEnds = 0;

    // Choose which GPU to run on, change this on a multi-GPU system.
    HANDLE_ERROR(cudaSetDevice(0));

    // TODO : OpenGL setup

    // Allocate memory
    sim::allocateMemory(&positionX, &positionY, &positionZ, &cellIds, &particleIds, &cellStarts, &cellEnds, particleCount);

    // Generate random positions
    sim::generateRandomPositions(positionX, positionY, positionZ, particleCount);

    // MAIN LOOP HERE - probably dictaded by glfw

    // while(true)
    //{
        // 1.
        sim::calculateNextFrame(positionX, positionY, positionZ, cellIds, particleIds, cellStarts, cellEnds, particleCount);

        // 2.
        // OpenGL render
    //}

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.

    cudaFree(positionX);
    cudaFree(positionY);
    cudaFree(positionZ);
    cudaFree(cellIds);
    cudaFree(particleIds);
    cudaFree(cellStarts);
    cudaFree(cellEnds);

    HANDLE_ERROR(cudaDeviceReset());

    return 0;
}