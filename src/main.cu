#include "simulation.cuh"
#include "objects.cuh"
#include "defines.cuh"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>

int main()
{
    unsigned int* cellIds = 0;
    unsigned int* particleIds = 0;
    unsigned int* cellStarts = 0;
    unsigned int* cellEnds = 0;


    // Choose which GPU to run on, change this on a multi-GPU system.
    HANDLE_ERROR(cudaSetDevice(0));

    // TODO : OpenGL setup

    // Allocate memory
    sim::allocateMemory(&cellIds, &particleIds, &cellStarts, &cellEnds, max_particles_count);

    // Generate random positions
    //sim::generateRandomPositions(positionX, positionY, positionZ, max_particles_count);
    sim::generateInitialPositionsInLayers(max_particles_count, 3);

    // MAIN LOOP HERE - probably dictaded by glfw

    // while(true)
    //{
        // 1.
        sim::calculateNextFrame(globalParticles.position.x, globalParticles.position.y, globalParticles.position.z, 
            cellIds, particleIds, cellStarts, cellEnds, real_particles_count);

        // 2.
        // OpenGL render
    //}

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.

    /*
    *   A lot more cleaning in structs and classes here
    */
    cudaFree(cellIds);
    cudaFree(particleIds);
    cudaFree(cellStarts);
    cudaFree(cellEnds);

    HANDLE_ERROR(cudaDeviceReset());

    return 0;
}