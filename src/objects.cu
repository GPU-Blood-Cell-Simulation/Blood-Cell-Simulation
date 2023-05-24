#include "objects.cuh"

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
