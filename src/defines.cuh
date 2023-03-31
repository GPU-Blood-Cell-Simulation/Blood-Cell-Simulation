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

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ));


#ifdef __INTELLISENSE__
void __syncthreads();
#endif



constexpr float3 dimension {100,100,100};

constexpr int cellWidth = 2;
constexpr int cellHeight = 2;
constexpr int cellDepth = 2;

// max particle count
constexpr int max_particles_count = 1e6;

// real particle count
// because some particle might been lost during initial scene casting
int real_particles_count = 0;

// ! this value should be determined experimentally !
// one frame simulation time span
constexpr float dt = 0.1f;

// initial particle velocity value
constexpr float v0 = 0.1f;

/// used in allocating new particles in array
unsigned int new_cell_index = 0;

/// PHYSICS CONST

// ! this value should be determined experimentally !
// Hooks law k factor from F = k*x
constexpr float k_sniff = 0.1f;

// ! this value should be determined experimentally !
// Damping factor 
constexpr float d_fact = 0.1f;


/// UTILITIES


__host__ __device__ float3 operator*(float a, float3 v)
{
	return make_float3(a * v.x, a * v.y, a * v.z);
}

__host__ __device__ float3 operator*(float3 a, float3 b)
{
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ float3 operator+(float3 a, float3 b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ float3 operator-(float3 a, float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ float3 operator/(float3 v, float a)
{
	return make_float3(v.x / a, v.y / a, v.z / a);
}

__host__ __device__ float dot(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ float3 cross(float3 u, float3 v)
{
	return make_float3(u.y * v.z - u.z * v.y,
		u.z * v.x - u.x * v.z,
		u.x * v.y - u.y * v.x);
}

__host__ __device__ float length(float3 v)
{
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__host__ __device__ float3 normalize(float3 v) // versor
{
	return v / sqrtf(dot(v, v));
}