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

// TODO: zmieniæ to, bo makra s¹ brzydkie

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ));

inline constexpr float windowWidth = 800;
inline constexpr float windowHeight = 800;

inline constexpr float width = 100.0f;
inline constexpr float height = 100.0f;
inline constexpr float depth = 100.0f;
inline constexpr int cellWidth = 2;
inline constexpr int cellHeight = 2;
inline constexpr int cellDepth = 2;

inline constexpr unsigned int particleCount = 500;

inline constexpr bool useLighting = false;