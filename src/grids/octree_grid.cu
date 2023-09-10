#include "octree_grid.cuh"
#include "grid_helpers.cuh"
#include "../defines.hpp"
#include "../utilities/cuda_handle_error.cuh"
#include "../utilities/math.cuh"
#include "math_functions.h"

#define EXPONENTIAL_MASK 0x7f800000
#define EXPONENTIAL_OFFSET 23
#define MANTIS_MASK 0x007fffff
#define MANTIS_OFFSET_LEFTSHIFTED 9

#define MANTIS_OR_MASK 0x00800000


#define MORTON_POSITION_MASK 0x07


struct positive_float_structure {

	unsigned int mantis;
	unsigned char exponent;
	
	positive_float_structure(float value) {
		unsigned int valueCasted = *(int*)&value;
		exponent = (valueCasted & EXPONENTIAL_MASK) >> EXPONENTIAL_OFFSET;
		mantis = valueCasted << MANTIS_OFFSET_LEFTSHIFTED;
	}
};

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
		unsigned int mask = 8 << 3 * levels;
		unsigned int realId = 0;

#pragma unroll
		for (int i = 3 * levels; i >= 0; i -= 3) {
			realId += (mask & currentCellId) >> i;
			realId *= 8;
		}

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

// transformed
__device__ unsigned int calculateCellForPosition(float3 position, float3 cellCenter)
{
	return ((cellCenter.z < position.z) << 2) | ((cellCenter.y < position.y) << 1) | (cellCenter.x < position.x);
}


__device__ float3 calculateChildCenter(float3 center, unsigned int childId, float3 childCellDimension)
{
	return center + make_float3(((childId & 1) ? childCellDimension.x : -childCellDimension.x),
		((childId & 2) ? childCellDimension.y : -childCellDimension.y), ((childId & 4) ? childCellDimension.z : -childCellDimension.z));
}

__device__ float3 calculateParentCenter(float3 center, unsigned int childId)
{
	return center + make_float3((!(childId & 1) ? center.x / 2 : -center.x / 2),
		(!(childId & 2) ? center.y / 2 : -center.y / 2), (!(childId & 4) ? center.z / 2 : -center.z / 2));
}

__device__ float3 calculateRayTValue(float3 origin, float3 inversedDirection, float3 argument)
{
	return (argument - origin) * inversedDirection;
}

__device__ float3 calculateLeafCellFromMorton(float3 cellDimension, float3 bounding, unsigned int mortonCode, unsigned int level) {
	
	float3 leafCell = make_float3(0, 0, 0);
#pragma unroll
	for (int i = 0; i < level - 1; i++) {
		unsigned int mask = mortonCode & 8;
		bounding = bounding / 2;
		if (mask & 1) {
			leafCell.x += bounding.x;
		}
		if (mask & 2) {
			leafCell.y += bounding.y;
		}
		if (mask & 4) {
			leafCell.z += bounding.z;
		}
		mortonCode >> 3;
	}
	return leafCell;
}

__device__ float3 calculateNeighbourLeafPos(float3 pos, float3 direction, float3 childCellSize, unsigned char bitChange)
{
	float3 newPos = pos;
	if (bitChange > 1) {
		if (direction.z < 0)
			newPos.z -= childCellSize.z;
		else
			newPos.z += childCellSize.z;
	}
	else if (bitChange) {
		if (direction.y < 0)
			newPos.y -= childCellSize.y;
		else
			newPos.y += childCellSize.y;
	}
	else {
		if (direction.x < 0)
			newPos.x -= childCellSize.x;
		else
			newPos.x += childCellSize.x;
	}
	return newPos;

}

__device__ void traverseGrid(float3 origin, float3 direction, float tmax, unsigned char* masks, unsigned int* treeData, const unsigned int maxLevel)
{
	// necessary parameters
	const float3 bounding = make_float3(width, height, depth);
	const unsigned int s_max = maxLevel;
	const unsigned int leafShift = (ldexp(1, 3 * (maxLevel - 1)) - 1) / 7; // (8^(maxL - 1) - 1)/7
	const float3 relativeOrigin = origin / bounding;
	const float3 directionSigns = make_float3(!(direction.x < 0), !(direction.y < 0), !(direction.z < 0)); // 1 plus or zero, 0 minus
	const float3 inversedDirection = make_float3(1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z);

	// values
	unsigned int parentId = 0; // root
	float3 pos = { 0.5f, 0.5f, 0.5f }; // initial pos is center of root
	unsigned int scale = s_max - 1;
	unsigned int childId = calculateCellForPosition(relativeOrigin, pos);
	unsigned int realChildId = 8 * parentId + childId + 1;
	float3 childCellSize = 0.5f*bounding;
	float3 tBegin = relativeOrigin;
	float3 tEnd = relativeOrigin;

	while (true) {

		// traversing down the stack
		if (scale > 0) {
			childCellSize = 0.5f*childCellSize;
			parentId = realChildId;
			pos = calculateChildCenter(pos, childId, childCellSize);
			childId = calculateCellForPosition(relativeOrigin, pos);
			unsigned int realChildId = 8 * parentId + childId + 1;
			
			if (!(masks[realChildId] & (1 << childId))) // empty cell
				break;

			//parentCellCenter = tempCenter; // maybe do not need tempCenter ??? 
			scale--;
			//scale = scale / 2;
			continue;
		}

		// compute intersections in current cell
		// TODO

		// calculate neighbour cell
		unsigned int leafMortonCode = treeData[realChildId - leafShift];

		//float3 cellBegining = calculateLeafCellFromMorton(scale, bounding, leafMortonCode, currentLevel);
		float3 cellBeginning = pos - make_float3(fmodf(pos.x, childCellSize.x),
			fmodf(pos.y, childCellSize.y), fmodf(pos.z, childCellSize.z));

		tBegin = tEnd;

		// maybe ifs instead of directionSigns ???
		tEnd = calculateRayTValue(relativeOrigin, inversedDirection, cellBeginning + directionSigns*childCellSize);
		float tMax = vmin(tEnd);

		bool changeParent = false;

		unsigned char bitChange = 0;
		// bit changing && should be + && is minus
		if (!(tMax > tEnd.x) && (childId & 1) && direction.x < 0) {
			changeParent = true;
		}
		else if (!(tMax > tEnd.x)) {
			bitChange = 1;
			if ((childId & 1) && direction.y < 0) {
				changeParent = true;
			}
		}
		else if (!(tMax > tEnd.x)) {
			bitChange = 2;
			if ((childId & 1) && direction.z < 0) {
				changeParent = true;
			}
		}
		
		if (changeParent) {
			// calculate new pos
			float3 newPos = calculateNeighbourLeafPos(pos, direction, childCellSize, bitChange);

			positive_float_structure posX(pos.x), posY(pos.y), posZ(pos.z);
			positive_float_structure newPosX(newPos.x), newPosY(newPos.y), newPosZ(newPos.z);

			unsigned char minBinaryPlacesPos = 127 - max(posX.exponent, max(posY.exponent, posZ.exponent));
			unsigned char minBinaryPlacesNewPos = 127 - max(newPosX.exponent, max(newPosY.exponent, newPosZ.exponent));

			if (minBinaryPlacesPos > minBinaryPlacesNewPos) {
				scale = s_max - minBinaryPlacesNewPos;
				childCellSize = ldexp(1, scale - s_max) * bounding;
				// childId =
				// parentId = 
			}
			else if (minBinaryPlacesPos < minBinaryPlacesNewPos) {

			}
			else { // the same, check mantisa

			}
		}
		else {

			// calculate new childId
			childId ^= 1 << bitChange;
			unsigned int realChildId = 8 * parentId + childId + 1;

			if (!(masks[realChildId] & (1 << childId))) // empty cell
				break;

			// calculate new pos
			pos = calculateNeighbourLeafPos(pos, direction, childCellSize, bitChange);
		}
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