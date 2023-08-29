#include "octree_grid.cuh"
#include "grid_helpers.cuh"
#include "../defines.hpp"
#include "../utilities/cuda_handle_error.cuh"
#include "../utilities/math.cuh"

#define EXPONENTIAL_MASK 0x7f800000
#define EXPONENTIAL_OFFSET 23
#define MANTIS_MASK 0x007fffff
#define MANTIS_OR_MASK 0x00800000

__global__ void calculateTreeLeafsCells(const unsigned int* cellIds, const unsigned int objectCount,
	const unsigned int cellCountX, const unsigned int cellCountY, const unsigned int cellCountZ,
	const unsigned int cellWidth, const unsigned int cellHeight, const unsigned int cellDepth, const unsigned int divisionCount
	, const float width, const float height, const float depth)
{
	unsigned int objectId = blockIdx.x * blockDim.x + threadIdx.x;
	if (objectId >= objectCount)
		return;

	unsigned int cellId = cellIds[objectId];

	const unsigned int idX = (cellId % cellWidth) / 2;
	const unsigned int idY = ((cellId / cellWidth) % cellHeight) / 2;
	const unsigned int idZ = ((cellId / cellWidth / cellHeight) % cellDepth) / 2;

	const float positionX = (2.0f * float(idX) + 0.5f) * cellWidth;
	const float positionY = (2.0f * float(idY) + 0.5f) * cellWidth;
	const float positionZ = (2.0f * float(idZ) + 0.5f) * cellDepth;

	const float lowerBoundX = float(idX) * cellWidth;
	const float lowerBoundY = float(idY) * cellHeight;
	const float lowerBoundZ = float(idZ) * cellDepth;
	
	float relativeLocationX = ((positionX - lowerBoundX) / width);
	float relativeLocationY = ((positionY - lowerBoundY) / height);
	float relativeLocationZ = ((positionZ - lowerBoundZ) / depth);

	const unsigned int locationCodeX = *(int*)&relativeLocationX;
	const unsigned int locationCodeY = *(int*)&relativeLocationY;
	const unsigned int locationCodeZ = *(int*)&relativeLocationZ;

	unsigned int counter = 0;
	

	// shift is 23 + (126 - exponent)
	unsigned int exponentShiftX = 149 - (locationCodeX & EXPONENTIAL_MASK) >> EXPONENTIAL_OFFSET;
	unsigned int exponentShiftY = 149 - (locationCodeY & EXPONENTIAL_MASK) >> EXPONENTIAL_OFFSET;
	unsigned int exponentShiftZ = 149 - (locationCodeZ & EXPONENTIAL_MASK) >> EXPONENTIAL_OFFSET;

	unsigned int fullMantisX = MANTIS_OR_MASK | (locationCodeX & MANTIS_MASK);
	unsigned int fullMantisY = MANTIS_OR_MASK | (locationCodeY & MANTIS_MASK);
	unsigned int fullMantisZ = MANTIS_OR_MASK | (locationCodeZ & MANTIS_MASK);

#pragma unroll
	while (counter++ < divisionCount) {
		

		__syncthreads();
	}

}

__global__ void calculateTreeFirstLevel(const float* positionX, const float* positionY, const float* positionZ,
	const unsigned int cellWidth, const unsigned int cellHeight, const unsigned int cellDepth, unsigned int levels,
	unsigned int* treeData, int16_t* masks, unsigned int objectCount)
{
	unsigned int objectId = blockIdx.x * blockDim.x + threadIdx.x;
	if (objectId >= objectCount)
		return;

	float positionX = positionX[objectId];
	float positionY = positionY[objectId];
	float positionZ = positionZ[objectId];



#pragma unroll
	for (int level = levels; level >= 0; --level)
	{

		__syncthreads();
	}
}

OctreeGrid::OctreeGrid(const unsigned int objectCount, const unsigned int levels)
{
	this->objectsCount = objectCount;
	this->levels = levels;

	cellWidth = width / pow(2, levels);
	cellHeight = height / pow(2, levels);
	cellDepth = depth / pow(2, levels);


	cellCountX = static_cast<unsigned int>(width / cellWidth);
	cellCountY = static_cast<unsigned int>(height / cellHeight);
	cellCountZ = static_cast<unsigned int>(depth / cellDepth);

	cellAmount = width / cellWidth * height / cellHeight * depth / cellDepth;
	HANDLE_ERROR(cudaMalloc((void**)&gridCellIds, objectsCount * sizeof(unsigned int)));
	HANDLE_ERROR(cudaMalloc((void**)&particleIds, objectsCount * sizeof(unsigned int)));

	HANDLE_ERROR(cudaMalloc((void**)&gridCellStarts, cellAmount * sizeof(unsigned int)));
	HANDLE_ERROR(cudaMalloc((void**)&gridCellEnds, cellAmount * sizeof(unsigned int)));

	treeNodesCount = (pow(8, levels) - 1) / 7;
	printf("Octree nodes count: %d\n", treeNodesCount);
	HANDLE_ERROR(cudaMalloc((void**)&treeData, treeNodesCount * sizeof(unsigned int)));
	HANDLE_ERROR(cudaMalloc((void**)&masks, treeNodesCount * sizeof(int16_t)));
}

void OctreeGrid::calculateGrid(const float* positionX, const float* positionY, const float* positionZ, unsigned int particleCount)
{
	gridHelpers::createUniformGrid(positionX, positionY, positionZ, gridCellIds, particleIds, cellWidth,
		cellHeight, cellDepth, gridCellStarts, gridCellEnds, particleCount);


}