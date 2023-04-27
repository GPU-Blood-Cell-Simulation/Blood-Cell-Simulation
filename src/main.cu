#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "simulation.cuh"
#include "defines.cuh"
#include "objects.cuh"
#include "window.h"

#include <curand.h>
#include <curand_kernel.h>

//// NVIDIA GPU selector for devices with multiple GPUs (e.g. laptops)
//extern "C"
//{
//    __declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
//}

int main()
{
    unsigned int* cellIds = 0;
    unsigned int* particleIds = 0;
    unsigned int* cellStarts = 0;
    unsigned int* cellEnds = 0;

    // Choose which GPU to run on, change this on a multi-GPU system.
    HANDLE_ERROR(cudaSetDevice(0));

    Window window(windowWidth, windowHeight);

    // Allocate memory
    particles particls(PARTICLE_COUNT);
    dipols corpscls = dipols(10);

    sim::allocateMemory(&cellIds, &particleIds, &cellStarts, &cellEnds, PARTICLE_COUNT);

    // Generate random positions
    sim::generateRandomPositions(particls, PARTICLE_COUNT);
    //sim::generateInitialPositionsInLayers(particls, corpscls, PARTICLE_COUNT, 3);

    // MAIN LOOP HERE - probably dictated by glfw

    while (!window.shouldClose())
    {
        window.clear();

        // Calculate particle positions using CUDA
        sim::calculateNextFrame(particls, corpscls, cellIds, particleIds, cellStarts, cellEnds, PARTICLE_COUNT);

        window.updateParticles(particls);
        window.calculateFPS();

        window.handleEvents();
    }

    // Cleanup
    cudaFree(particls.position.x);
    cudaFree(particls.position.y);
    cudaFree(particls.position.z);
    cudaFree(particls.velocity.x);
    cudaFree(particls.velocity.y);
    cudaFree(particls.velocity.z);
    cudaFree(particls.force.x);
    cudaFree(particls.force.y);
    cudaFree(particls.force.z);
    cudaFree(cellIds);
    cudaFree(particleIds);
    cudaFree(cellStarts);
    cudaFree(cellEnds);

    HANDLE_ERROR(cudaDeviceReset());

    return 0;
}