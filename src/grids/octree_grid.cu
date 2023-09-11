#include "octree_grid.cuh"
#include "grid_helpers.cuh"
#include "../defines.hpp"
#include "../utilities/cuda_handle_error.cuh"
#include "../utilities/math.cuh"
#include "math_functions.h"

__device__ unsigned int partEveryByteByTwo(unsigned int n)
{
	n = (n ^ (n << 16)) & 0xff0000ff;
	n = (n ^ (n << 8)) & 0x0300f00f;
	n = (n ^ (n << 4)) & 0x030c30c3;
	n = (n ^ (n << 2)) & 0x09249249;
	return n;
}

__device__ unsigned int calculateMortonCodeIdForCell(unsigned int xId, unsigned int yId, unsigned int zId)
{
	return (partEveryByteByTwo(zId) << 2) | (partEveryByteByTwo(yId) << 1) | partEveryByteByTwo(xId);
}

__global__ void calculateCellIdKernel(const float* positionX, const float* positionY, const float* positionZ,
	unsigned int* cellIds, unsigned int* particleIds, const unsigned int particleCount,
	unsigned int cellWidth, unsigned int cellHeight, unsigned int cellDepth)
{
	unsigned int particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId >= particleCount)
		return;

	unsigned int cellId = calculateMortonCodeIdForCell(unsigned int(positionX[particleId] / cellWidth),
		unsigned int(positionY[particleId] / cellHeight), unsigned int(positionZ[particleId] / cellDepth));

	particleIds[particleId] = particleId;
	cellIds[particleId] = cellId;
}

__device__ unsigned int calculateCellIdFromMorton(unsigned int mortonCode, unsigned int levels)
{
	unsigned int mask = 8 << 3 * levels;
	unsigned int realId = 0;

#pragma unroll
	for (int i = 3 * levels; i >= 0; i -= 3) {
		realId += (mask & mortonCode) >> i;
		realId *= 8;
	}
	return realId;
}

__global__ void calculateCellStarts(const unsigned int* cellIds, const unsigned int* particleIds,
	unsigned int* treeData, unsigned int cellCount, unsigned int levels)
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= cellCount)
		return;

	unsigned int currentCellId = cellIds[id];

	if (id == 0 || cellIds[id - 1] >> 3 < currentCellId >> 3)
	{
		levels -= 2;
		unsigned int realId = calculateCellIdFromMorton(currentCellId, levels);
		unsigned int index = realId - (unsigned int)(pow(8, levels) - 1) / 7;
		treeData[index] = id;
	}
}

void createOctreeGridData(const float* positionX, const float* positionY, const float* positionZ, unsigned int* gridCellIds,
	unsigned int* particleIds, float cellWidth, float cellHeight, float cellDepth, /*unsigned int* gridCelLStarts,
	unsigned int* gridCellEnds,*/ unsigned int objectCount)
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
	calculateCellStarts(gridCellIds, particleIds, treeData, objectCount, levels);
	/*gridHelpers::calculateStartAndEndOfCellKernel << <blocks, threadsPerBlock >> >
		(positionX, positionY, positionZ, gridCellIds, particleIds, gridCellStarts, gridCellEnds, objectCount);*/
}




__global__ void calculateTreeLeafsCells(const unsigned int* cellIds, const unsigned int objectCount,
	const unsigned int cellCountX, const unsigned int cellCountY, const unsigned int cellCountZ,
	const unsigned int cellWidth, const unsigned int cellHeight, const unsigned int cellDepth, const unsigned int divisionCount
	, const float width, const float height, const float depth, unsigned char* masks)
{
	unsigned int objectId = blockIdx.x * blockDim.x + threadIdx.x;
	if (objectId >= objectCount)
		return;

	unsigned int cellMortonCodeId = cellIds[objectId];
	unsigned int currentCellAbsoluteId = cellMortonCodeId + (pow(8, divisionCount) - 1) / 7;
	unsigned int counter = 0;


#pragma unroll
	while (counter++ < divisionCount - 1) {
		unsigned int parentId = currentCellAbsoluteId >> 3;
		unsigned char childFillMask = 1 << (currentCellAbsoluteId & MORTON_POSITION_MASK);

		if (!(masks[parentId] & childFillMask))
			atomicOr_system(masks + parentId, childFillMask);
			masks[parentId] |= childFillMask;
		
		currentCellAbsoluteId = parentId;
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

	//HANDLE_ERROR(cudaMalloc((void**)&gridCellStarts, cellAmount * sizeof(unsigned int)));
	//HANDLE_ERROR(cudaMalloc((void**)&gridCellEnds, cellAmount * sizeof(unsigned int)));

	treeNodesCount = (pow(8, levels) - 1) / 7;
	printf("Octree nodes count: %d\n", treeNodesCount);
	HANDLE_ERROR(cudaMalloc((void**)&masks, treeNodesCount * sizeof(unsigned char)));
	
	lastNonLeafLayerCount = pow(8, levels - 2);
	HANDLE_ERROR(cudaMalloc((void**)&treeData, lastNonLeafLayerCount * sizeof(unsigned int)));
}

void OctreeGrid::calculateGrid(const float* positionX, const float* positionY, const float* positionZ, unsigned int particleCount)
{
	createOctreeGridData(positionX, positionY, positionZ, gridCellIds, particleIds, cellWidth,
		cellHeight, cellDepth, /*gridCellStarts, gridCellEnds, */particleCount);


}