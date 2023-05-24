#include "objects.cuh"
#include "defines.cuh"
#include "utilities.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// cudaVec3
void cudaVec3::createVec(int n)
{
    size = n;
	// allocate 
	HANDLE_ERROR(cudaMalloc((void**)&x, n * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&y, n * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&z, n * sizeof(float)));
}

void cudaVec3::freeVec()
{
	cudaFree(x);
	cudaFree(y);
	cudaFree(z);
}

__device__ float3 cudaVec3::get(int index)
{
	return make_float3(x[index], y[index], z[index]);
}

__device__ void cudaVec3::set(int index, float3 v)
{
	x[index] = v.x;
	y[index] = v.y;
	z[index] = v.z;
}

__device__ void cudaVec3::add(int index, float3 v)
{
	x[index] += v.x;
	y[index] += v.y;
	z[index] += v.z;
}

