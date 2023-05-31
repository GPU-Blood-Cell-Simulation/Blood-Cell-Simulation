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


inline constexpr float windowWidth = 800;
inline constexpr float windowHeight = 800;

inline constexpr float width = 100.0f;
inline constexpr float height = 100.0f;
inline constexpr float depth = 100.0f;
//constexpr float3 dimension {100,100,100};


// cylinder model data:
// min_x = -1.00156, max_x = 0.998437, min_y = -0.130239, max_y = 5.14687
inline constexpr float cylinderRadius = 1.0f;
inline constexpr float cylinderHeight = 5.27716f; // scale = glm::vec3(width / 2, 2 * width, width / 2);
inline constexpr float cylinderScaleX = width / 2;
inline constexpr float cylinderScaleY = 2 * height;
inline constexpr float cylinderScaleZ = depth / 2;

inline constexpr int cellWidth = 5;
inline constexpr int cellHeight = 5;
inline constexpr int cellDepth = 5;

// 96 = 3*2*4*4
// three layers of 2 particle dipoles, each layer is 4x4
inline constexpr unsigned int PARTICLE_COUNT = 200; 

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

// Lighting
inline constexpr bool useLighting = true;

// Camera movement constants
inline constexpr float cameraMovementSpeed = width / 100;
inline constexpr float cameraRotationSpeed = 0.02f;
