#include "uniform_grid.cuh"
#include "../defines.hpp"
#include "../utilities/cuda_handle_error.cuh"

#include <cstdio>
#include <cstdlib>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#pragma region Helper kernels

__global__ void calculateCellIdKernel(const float* positionX, const float* positionY, const float* positionZ,
	unsigned int* cellIds, unsigned int* particleIds, const unsigned int particleCount)
{
	unsigned int particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId >= particleCount)
		return;

	unsigned int cellId =
		static_cast<unsigned int>(positionZ[particleId] / cellDepth) * static_cast<unsigned int>(width / cellWidth) * static_cast<unsigned int>(height / cellHeight) +
		static_cast<unsigned int>(positionY[particleId] / cellHeight) * static_cast<unsigned int>(width / cellWidth) +
		static_cast<unsigned int>(positionX[particleId] / cellWidth);

	// Debug
	/*if (cellId >= 9261)
		printf("Error, cellId: %d\n", cellId);*/

	//printf("id: %d, cellId: %d\n", particleId, cellId);

	particleIds[particleId] = particleId;
	cellIds[particleId] = cellId;

}

__global__ void calculateStartAndEndOfCellKernel(const float* positionX, const float* positionY, const float* positionZ,
	const unsigned int* cellIds, const unsigned int* particleIds,
	unsigned int* cellStarts, unsigned int* cellEnds, unsigned int particleCount)
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= particleCount)
		return;

	unsigned int currentCellId = cellIds[id];

	// Check if the previous cell id was different - it would mean we found the start of a cell
	if (id > 0 && currentCellId != cellIds[id - 1])
	{
		cellStarts[currentCellId] = id;
	}

	// Check if the next cell id was different - it would mean we found the end of a cell
	if (id < particleCount - 1 && currentCellId != cellIds[id + 1])
	{
		cellEnds[currentCellId] = id;
	}

	if (id == 0)
	{
		cellStarts[cellIds[0]] = 0;
	}
	if (id == particleCount - 1)
	{
		cellStarts[cellIds[particleCount - 1]] = particleCount - 1;
	}
}

#pragma endregion

// Allocate GPU buffers for the index buffers
UniformGrid::UniformGrid(const unsigned int particleCount)
{
	HANDLE_ERROR(cudaMalloc((void**)&cellIds, particleCount * sizeof(unsigned int)));
	HANDLE_ERROR(cudaMalloc((void**)&particleIds, particleCount * sizeof(unsigned int)));

	HANDLE_ERROR(cudaMalloc((void**)&cellStarts, width / cellWidth * height / cellHeight * depth / cellDepth * sizeof(unsigned int)));
	HANDLE_ERROR(cudaMalloc((void**)&cellEnds, width / cellWidth * height / cellHeight * depth / cellDepth * sizeof(unsigned int)));
}

UniformGrid::UniformGrid(const UniformGrid& other) : isCopy(true), cellIds(other.cellIds), particleIds(other.particleIds),
	cellStarts(other.cellStarts), cellEnds(other.cellEnds) {}

UniformGrid::~UniformGrid()
{
	if (!isCopy)
	{
		HANDLE_ERROR(cudaFree(cellIds));
		HANDLE_ERROR(cudaFree(particleIds));
		HANDLE_ERROR(cudaFree(cellStarts));
		HANDLE_ERROR(cudaFree(cellEnds));
	}
	
}

void UniformGrid::calculateGrid(const Particles& particles)
{
	calculateGrid(particles.position.x, particles.position.y, particles.position.z, PARTICLE_COUNT);
}

void UniformGrid::calculateGrid(const float* positionX, const float* positionY, const float* positionZ, unsigned int objectsCount)
{
	// Calculate launch parameters

	const int threadsPerBlock = objectsCount > 1024 ? 1024 : objectsCount;
	const int blocks = (objectsCount + threadsPerBlock - 1) / threadsPerBlock;

	// 1. Calculate cell id for every particle and store as pair (cell id, particle id) in two buffers
	calculateCellIdKernel << <blocks, threadsPerBlock >> >
		(positionX, positionY, positionZ, cellIds, particleIds, objectsCount);

	// 2. Sort particle ids by cell id

	thrust::device_ptr<unsigned int> keys = thrust::device_pointer_cast<unsigned int>(cellIds);
	thrust::device_ptr<unsigned int> values = thrust::device_pointer_cast<unsigned int>(particleIds);

	thrust::stable_sort_by_key(keys, keys + objectsCount, values);

	// 3. Find the start and end of every cell

	calculateStartAndEndOfCellKernel << <blocks, threadsPerBlock >> >
		(positionX, positionY, positionZ, cellIds, particleIds, cellStarts, cellEnds, objectsCount);
}