#include "simulation.cuh"

struct cudaVec3
{
	float* x;
	float* y;
	float* z;
	int size;

	cudaVec3(int n);

	__device__ float3 get(int index);
	__device__ void set(int index, float3 v);
	__device__ void add(int index, float3 v);
};
