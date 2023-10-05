#include "octree_grid.cuh"
#include "../defines.hpp"
#include "../utilities/cuda_handle_error.cuh"
#include "../utilities/math.cuh"
#include <cstdio>
#include <cstdlib>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
////
////#ifndef __CUDA_ARCH__
////#define __CUDA_ARCH__
////#endif
//#include "sm_60_atomic_functions.h"


#define MORTON_POSITION_MASK 0x07

#ifdef __INTELLISENSE__
template<typename T>
void atomicAdd(T*, T);
template<typename T>
void atomicOr(T*, T);
#endif

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

__global__ void calculateOctreeCellIdKernel(const float* positionX, const float* positionY, const float* positionZ,
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

//__device__ unsigned int calculateCellIdFromMorton(unsigned int mortonCode, unsigned int levels)
//{
//	unsigned int mask = 8 << 3 * levels;
//	unsigned int realId = 0;
//
//#pragma unroll
//	for (int i = 3 * levels; i >= 0; i -= 3) {
//		realId += (mask & mortonCode) >> i;
//		realId *= 8;
//	}
//	return realId;
//}

__global__ void calculateCellStarts(const unsigned int* cellIds, const unsigned int* particleIds,
	unsigned int* treeData, unsigned int cellCount)
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= cellCount)
		return;

	unsigned int currentCellId = cellIds[id];

	if (id == 0 || cellIds[id - 1] >> 3 < currentCellId >> 3)
	{
		treeData[currentCellId] = id;
	}
}



__global__ void calculateTreeLeafsCells(const unsigned int* cellIds, const unsigned int objectCount, const unsigned int levels
	, unsigned char* masks, unsigned char* shifts)
{
	unsigned int objectId = blockIdx.x * blockDim.x + threadIdx.x;
	if (objectId >= objectCount)
		return;

	unsigned int cellMortonCodeId = cellIds[objectId];
	//unsigned int currentCellAbsoluteId = cellMortonCodeId + (ldexp((double)1, 3 * (levels - 1)) - 1) / 7; // ( 8^(levels - 1) - 1 ) / 7
	unsigned int currentLevel = levels;

#pragma unroll
	while (--currentLevel > 0) {
		unsigned int levelMask = MORTON_POSITION_MASK << (currentLevel - 1)*3;
		unsigned int parentId = cellMortonCodeId & !(levelMask);
		unsigned char childFillMask = 1 << levelMask;

		unsigned int realParentId = shifts[currentLevel] + parentId;
		if (!(masks[realParentId] & childFillMask)) {
			unsigned char* memPtr = masks + realParentId;
			atomicOr((unsigned int*)memPtr, (unsigned int)childFillMask); //masks[parentId] |= childFillMask;
		}

		cellMortonCodeId = parentId;
	}

}



void createOctreeGridData(const float* positionX, const float* positionY, const float* positionZ, unsigned int* gridCellIds,
	unsigned int* particleIds, float cellWidth, float cellHeight, float cellDepth, unsigned int* treeData, unsigned char* masks,
	unsigned int* shifts, unsigned int levels, unsigned int objectCount)
{
	// Calculate launch parameters

	const int threadsPerBlock = objectCount > 1024 ? 1024 : objectCount;
	const int blocks = (objectCount + threadsPerBlock - 1) / threadsPerBlock;

	// 1. Calculate cell id for every particle and store as pair (cell id, particle id) in two buffers
	calculateOctreeCellIdKernel << <blocks, threadsPerBlock >> >
		(positionX, positionY, positionZ, gridCellIds, particleIds, objectCount, cellWidth, cellHeight, cellDepth);

	// 2. Sort particle ids by cell id

	thrust::device_ptr<unsigned int> keys = thrust::device_pointer_cast<unsigned int>(gridCellIds);
	thrust::device_ptr<unsigned int> values = thrust::device_pointer_cast<unsigned int>(particleIds);

	thrust::stable_sort_by_key(keys, keys + objectCount, values);

	// 3. Find the start of every cell
	calculateCellStarts << <blocks, threadsPerBlock >> > (gridCellIds, particleIds, treeData, objectCount);


	calculateTreeLeafsCells << <blocks, threadsPerBlock >> > (gridCellIds, objectCount, levels, masks, shifts);
}

OctreeGrid::OctreeGrid(const OctreeGrid& other) : isCopy(true), gridCellIds(other.gridCellIds), particleIds(other.particleIds),
 cellCountX(other.cellCountX), cellCountY(other.cellCountY), cellCountZ(other.cellCountZ), cellWidth(other.cellWidth), cellHeight(other.cellHeight), 
	cellDepth(other.cellDepth), objectsCount(other.objectsCount), cellAmount(other.cellAmount), levels(other.levels), treeNodesCount(other.treeNodesCount),
	leafLayerCount(other.leafLayerCount), treeData(other.treeData), masks(other.masks)
{}

OctreeGrid::~OctreeGrid()
{
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
	HANDLE_ERROR(cudaMalloc((void**)&shifts, levels * sizeof(unsigned int)));

	unsigned int* shiftsHost = new unsigned int[levels];
	for(int i = 0, int index = 0; i < levels; ++i, index = index*8 + 1) {
		shiftsHost = index;
	}
	HANDLE_ERROR(cudaMemcpy(shifts, shiftsHost,  levels*sizeof(unsigned int), cudaMemcpyHostToDevice));
	//HANDLE_ERROR(cudaMalloc((void**)&gridCellStarts, cellAmount * sizeof(unsigned int)));
	//HANDLE_ERROR(cudaMalloc((void**)&gridCellEnds, cellAmount * sizeof(unsigned int)));

	treeNodesCount = (pow(8, levels) - 1) / 7;
	leafLayerCount = pow(8, levels - 1);
	printf("Octree nodes count: %d\n", treeNodesCount);
	HANDLE_ERROR(cudaMalloc((void**)&masks, (treeNodesCount - leafLayerCount )* sizeof(unsigned char)));
	
	HANDLE_ERROR(cudaMalloc((void**)&treeData, leafLayerCount * sizeof(unsigned int)));
}

void OctreeGrid::calculateGrid(const float* positionX, const float* positionY, const float* positionZ, unsigned int particleCount)
{
	createOctreeGridData(positionX, positionY, positionZ, gridCellIds, particleIds, cellWidth,
		cellHeight, cellDepth, treeData, masks, shifts, levels, particleCount);
}