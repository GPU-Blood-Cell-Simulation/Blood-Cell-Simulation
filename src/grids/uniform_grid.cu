#include "uniform_grid.cuh"
#include "../defines.hpp"
#include "../utilities/cuda_handle_error.cuh"

#include <cstdio>
#include <cstdlib>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>


#define max(a,b) ( a > b ? a : b)
#define min(a,b) ( a > b ? b : a)

__device__ unsigned int calculateIdForCell(float x, float y, float z, unsigned int cellWidth, unsigned int cellHeight, unsigned int cellDepth)
{
	if (x < 0 || x > width || y < 0 || y > height || z < 0 || z > depth) {
		printf("Position out of grid bounds: (%f, %f, %f)\n", x, y, z);
	}

	// should we clamp x,y,z if out of bounds?
	return
		static_cast<unsigned int>(min(width, max(0, z / cellDepth))) * static_cast<unsigned int>(width / cellWidth) * static_cast<unsigned int>(height / cellHeight) +
		static_cast<unsigned int>(min(height, max(0, y / cellHeight))) * static_cast<unsigned int>(width / cellWidth) +
		static_cast<unsigned int>(min(depth, max(0, x / cellWidth)));
}

__global__ void calculateCellIdKernel(const float* positionX, const float* positionY, const float* positionZ,
	unsigned int* cellIds, unsigned int* particleIds, const unsigned int particleCount,
	unsigned int cellWidth, unsigned int cellHeight, unsigned int cellDepth)
{
	unsigned int particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId >= particleCount)
		return;

	unsigned int cellId = calculateIdForCell(positionX[particleId], positionY[particleId], positionZ[particleId], cellWidth, cellHeight, cellDepth);

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

void createUniformGrid(const float* positionX, const float* positionY, const float* positionZ, unsigned int* gridCellIds,
	unsigned int* particleIds, float cellWidth, float cellHeight, float cellDepth, unsigned int* gridCellStarts,
	unsigned int* gridCellEnds, unsigned int objectCount)
{
	// Calculate launch parameters

	const int threadsPerBlock = objectCount > 1024 ? 1024 : objectCount;
	const int blocks = (objectCount + threadsPerBlock - 1) / threadsPerBlock;

	// 1. Calculate cell id for every particle and store as pair (cell id, particle id) in two buffers
	calculateCellIdKernel << <blocks, threadsPerBlock >> >
		(positionX, positionY, positionZ, gridCellIds, particleIds, objectCount, cellWidth, cellHeight, cellDepth);

	// 2. Sort particle ids by cell id

	thrust::device_ptr<unsigned int> keys = thrust::device_pointer_cast<unsigned int>(gridCellIds);
	thrust::device_ptr<unsigned int> values = thrust::device_pointer_cast<unsigned int>(particleIds);

	thrust::stable_sort_by_key(keys, keys + objectCount, values);

	// 3. Find the start and end of every cell

	calculateStartAndEndOfCellKernel << <blocks, threadsPerBlock >> >
		(positionX, positionY, positionZ, gridCellIds, particleIds, gridCellStarts, gridCellEnds, objectCount);
}

// Allocate GPU buffers for the index buffers
UniformGrid::UniformGrid(const unsigned int objectsCount, unsigned int cellWidth, unsigned int cellHeight, unsigned int cellDepth)
{
	this->cellWidth = cellWidth;
	this->cellHeight= cellHeight;
	this->cellDepth = cellDepth;
	this->objectsCount = objectsCount;
	cellCountX = static_cast<unsigned int>(width / cellWidth);
	cellCountY = static_cast<unsigned int>(height / cellHeight);
	cellCountZ = static_cast<unsigned int>(depth / cellDepth);

	cellAmount = width / cellWidth * height / cellHeight * depth / cellDepth;
	HANDLE_ERROR(cudaMalloc((void**)&gridCellIds, objectsCount* sizeof(unsigned int)));
	HANDLE_ERROR(cudaMalloc((void**)&particleIds, objectsCount * sizeof(unsigned int)));

	HANDLE_ERROR(cudaMalloc((void**)&gridCellStarts, cellAmount * sizeof(unsigned int)));
	HANDLE_ERROR(cudaMalloc((void**)&gridCellEnds, cellAmount * sizeof(unsigned int)));
}

UniformGrid::UniformGrid(const UniformGrid& other) : isCopy(true), gridCellIds(other.gridCellIds), particleIds(other.particleIds),
	gridCellStarts(other.gridCellStarts), gridCellEnds(other.gridCellEnds), cellCountX(other.cellCountX), cellCountY(other.cellCountY), cellCountZ(other.cellCountZ),
	cellWidth(other.cellWidth), cellHeight(other.cellHeight), cellDepth(other.cellDepth), objectsCount(other.objectsCount), cellAmount(other.cellAmount)
{}

UniformGrid::~UniformGrid()
{
	if (!isCopy)
	{
		HANDLE_ERROR(cudaFree(gridCellIds));
		HANDLE_ERROR(cudaFree(particleIds));
		HANDLE_ERROR(cudaFree(gridCellStarts));
		HANDLE_ERROR(cudaFree(gridCellEnds));
	}
}

void UniformGrid::calculateGrid(const float* positionX, const float* positionY, const float* positionZ, unsigned int objectCount)
{
	createUniformGrid(positionX, positionY, positionZ, gridCellIds, 
		particleIds, cellWidth, cellDepth, cellHeight, gridCellStarts, gridCellEnds, objectCount);
}

__device__ unsigned int UniformGrid::calculateCellId(float3 position)
{
	return calculateIdForCell(position.x, position.y, position.z, cellWidth, cellHeight, cellDepth);
}