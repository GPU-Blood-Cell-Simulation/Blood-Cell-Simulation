#pragma once

#include <cstdio>
#include <cstdlib>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// A handy function for easy error checking
static void HandleError(cudaError_t err, const char* file, int line);

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

// TODO: zmieni� to, bo makra s� brzydkie

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ));

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

inline constexpr float windowWidth = 800;
inline constexpr float windowHeight = 800;

constexpr float3 dimension {100,100,100};

inline constexpr int cellWidth = 2;
inline constexpr int cellHeight = 2;
inline constexpr int cellDepth = 2;

inline constexpr unsigned int PARTICLE_COUNT = 60000;

// ! this value should be determined experimentally !
// one frame simulation time span
__device__ constexpr float dt = 0.1f;

// initial particle velocity value
__device__ constexpr float v0 = 0.1f;

/// used in allocating new particles in array
unsigned int new_cell_index = 0;

/// PHYSICS CONST

// ! this value should be determined experimentally !
// Hooks law k factor from F = k*x
__device__ constexpr float k_sniff = 0.1f;

// ! this value should be determined experimentally !
// Damping factor 
__device__ constexpr float d_fact = 0.1f;

// Lighting does not work yet
inline constexpr bool useLighting = false;