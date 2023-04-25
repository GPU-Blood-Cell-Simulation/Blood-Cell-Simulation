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

// TODO: zmienić to, bo makra są brzydkie

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ));

#ifdef __INTELLISENSE__
void __syncthreads();
#endif


#ifndef GLBL_N_CNSTS
inline constexpr float windowWidth = 800;
inline constexpr float windowHeight = 800;

inline constexpr float width = 100.0f;
inline constexpr float height = 100.0f;
inline constexpr float depth = 100.0f;
//constexpr float3 dimension {100,100,100};

inline constexpr int cellWidth = 2;
inline constexpr int cellHeight = 2;
inline constexpr int cellDepth = 2;

// 96 = 3*2*4*4
// three layers of 2 particle dipoles, each layer is 4x4
inline constexpr unsigned int PARTICLE_COUNT = 90; 

// ! this value should be determined experimentally !
// one frame simulation time span
inline constexpr float dt = 0.1f;

// initial particle velocity value
inline constexpr float v0 = 0.1f;

/// used in allocating new particles in array
//inline unsigned int new_cell_index = 0;

/// PHYSICS CONST

// ! this value should be determined experimentally !
// Hooks law k factor from F = k*x
inline constexpr float k_sniff = 0.1f;

// ! this value should be determined experimentally !
// Damping factor 
inline constexpr float d_fact = 0.1f;


inline constexpr bool useLighting = false;
#define GLBL_N_CNSTS
#endif
