#include "uniform_grid.cuh"
#include "../defines.hpp"
#include "../utilities/cuda_handle_error.cuh"

#include <cstdio>
#include <cstdlib>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "grid_helpers.cuh"

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
	gridHelpers::createUniformGrid(positionX, positionY, positionZ, gridCellIds, 
		particleIds, cellWidth, cellDepth, cellHeight, gridCellStarts, gridCellEnds, objectCount);
}

__device__ unsigned int UniformGrid::calculateCellId(float3 position)
{
	return gridHelpers::calculateIdForCell(position.x, position.y, position.z, cellWidth, cellHeight, cellDepth);
}