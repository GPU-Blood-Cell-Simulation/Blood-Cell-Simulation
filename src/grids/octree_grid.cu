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
T atomicOr(T*, T);
template<typename T>
T atomicCAS(T*, T, T);
unsigned int __byte_perm(unsigned int, unsigned int, unsigned int);
void __syncthreads();
#endif

/*
__device__ inline char atomicCAS(char* address, char expected, char desired) {
	size_t long_address_modulo = (size_t)address & 3;
	auto* base_address = (unsigned int*)((char*)address - long_address_modulo);
	unsigned int selectors[] = { 0x3214, 0x3240, 0x3410, 0x4210 };

	unsigned int sel = selectors[long_address_modulo];
	unsigned int long_old, long_assumed, long_val, replacement;
	char old;

	long_val = (unsigned int)desired;
	long_old = *base_address;
	do {
		long_assumed = long_old;
		replacement = __byte_perm(long_old, long_val, sel);
		long_old = atomicCAS(base_address, long_assumed, replacement);
		old = (char)((long_old >> (long_address_modulo * 8)) & 0x000000ff);
	} while (expected == old && long_assumed != long_old);

	return old;
}*/
//__device__
//static inline
//uint8_t
//atomicCAS(uint8_t* const address,
//	uint8_t   const compare,
//	uint8_t   const value)
//{
//	// Determine where in a byte-aligned 32-bit range our address of 8 bits occurs.
//	uint8_t    const     longAddressModulo = reinterpret_cast<size_t>(address) & 0x3;
//	// Determine the base address of the byte-aligned 32-bit range that contains our address of 8 bits.
//	uint32_t* const     baseAddress = reinterpret_cast<uint32_t*>(address - longAddressModulo);
//	uint32_t   constexpr byteSelection[] = { 0x3214, 0x3240, 0x3410, 0x4210 }; // The byte position we work on is '4'.
//	uint32_t   const     byteSelector = byteSelection[longAddressModulo];
//	uint32_t   const     longCompare = compare;
//	uint32_t   const     longValue = value;
//	uint32_t             longOldValue = *baseAddress;
//	uint32_t             longAssumed;
//	uint8_t              oldValue;
//
//	do
//	{
//		// Select bytes from the old value and new value to construct a 32-bit value to use.
//		uint32_t const replacement = __byte_perm(longOldValue, longValue, byteSelector);
//		uint32_t const comparison = __byte_perm(longOldValue, longCompare, byteSelector);
//
//		longAssumed = longOldValue;
//		// Use 32-bit atomicCAS() to try and set the 8-bits we care about.
//		longOldValue = ::atomicCAS(baseAddress, comparison, replacement);
//		// Grab the 8-bit portion we care about from the old value at address.
//		oldValue = (longOldValue >> (8 * longAddressModulo)) & 0xFF;
//	} while (compare == oldValue && longAssumed != longOldValue); // Repeat until other three 8-bit values stabilize.
//
//	return oldValue;
//}

__device__ static inline unsigned char atomicCAS(char* address, char expected, char desired) {
	size_t long_address_modulo = (size_t)address & 3;
	auto* base_address = (unsigned int*)((char*)address - long_address_modulo);
	unsigned int selectors[] = { 0x3214, 0x3240, 0x3410, 0x4210 };

	unsigned int sel = selectors[long_address_modulo];
	unsigned int long_old, long_assumed, long_val, replacement;
	char old;

	long_val = (unsigned int)desired;
	long_old = *base_address;
	do {
		long_assumed = long_old;
		replacement = __byte_perm(long_old, long_val, sel);
		long_old = atomicCAS(base_address, long_assumed, replacement);
		old = (char)((long_old >> (long_address_modulo * 8)) & 0x000000ff);
	} while (expected == old && long_assumed != long_old);

	return old;
}

__device__ inline unsigned char atomicOr(unsigned char* address, unsigned char value) {
	unsigned char previousValue = *address;
	return atomicCAS((char*)address, (char)previousValue, char(previousValue | value));
}


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


__device__ char* int_bin_print(unsigned int v )
{
	char str[33];
	for (int i = 31; i >= 0; --i) {
		str[31 - i] = (v >> i) & 1 ? '1' : '0';
	}
	str[32] = '\0';
	return str;
}

__device__ char* char_bin_print(unsigned char v)
{
	char str[9];
	for (int i = 7; i >= 0; --i) {
		str[7 - i] = (v >> i) & 1 ? '1' : '0';
	}
	str[8] = '\0';
	return str;
}

__device__ bool getByte(unsigned int v, int byteNum)
{
	return (v >> byteNum) & 1;
}

__global__ void calculateOctreeCellIdKernel(const float* positionX, const float* positionY, const float* positionZ,
	unsigned int* cellIds, unsigned int* particleIds, const unsigned int particleCount,
	float cellWidth, float cellHeight, float cellDepth)
{
	unsigned int particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId >= particleCount)
		return;

	unsigned int cellId = calculateMortonCodeIdForCell(unsigned int(positionX[particleId] / cellWidth),
		unsigned int(positionY[particleId] / cellHeight), unsigned int(positionZ[particleId] / cellDepth));

	//printf("[particle %d] x %d of %d, y %d of %d, z %d of %d, cellId %d, cellMortonCode %s\n", particleId, unsigned int(positionX[particleId] / cellWidth), unsigned int(width / cellWidth)
	//	, unsigned int(positionY[particleId] / cellHeight), unsigned int(height / cellHeight), unsigned int(positionZ[particleId] / cellDepth), unsigned int(depth / cellDepth), cellId, int_bin_print(cellId));
	//printf("[particle %d] x %.5f y %.5f z %.5f , cellId %d, cellMortonCode %s\n", particleId, positionX[particleId] / width, positionY[particleId] / height, positionZ[particleId] / depth, cellId, int_bin_print(cellId));
	/*printf("[particle %d] x %.5f y %.5f z %.5f , cellId %d, cellMortonCode %d%d%d %d%d%d %d%d%d %d%d%d %d%d%d\n", particleId,  positionX[particleId]/width, positionY[particleId]/height, positionZ[particleId]/depth, particleId, cellId,
		getByte(cellId, 0), getByte(cellId, 1), getByte(cellId, 2), getByte(cellId, 3), getByte(cellId, 4), getByte(cellId, 5), getByte(cellId, 6), getByte(cellId, 7), getByte(cellId, 8), getByte(cellId, 9), getByte(cellId, 10), 
		getByte(cellId, 11), getByte(cellId, 12), getByte(cellId, 13), getByte(cellId, 14), getByte(cellId, 15));*/

	particleIds[particleId] = particleId;
	cellIds[particleId] = cellId;
}

__global__ void calculateCellStarts(const unsigned int* cellIds, const unsigned int* particleIds,
	unsigned int* treeData, unsigned int cellCount)
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= cellCount)
		return;

	unsigned int currentCellId = cellIds[id];

	if (id == 0 || cellIds[id - 1] != currentCellId )
	{
		treeData[currentCellId] = 0x80000000 | id;
	}
}

__global__ void calculateTreeLeafsCells(const unsigned int* cellIds, const unsigned int* gridCellStarts, const unsigned int objectCount, const unsigned int levels
	, unsigned char* masks, unsigned int* shifts)
{
	unsigned int objectId = blockIdx.x * blockDim.x + threadIdx.x;
	if (objectId >= objectCount)
		return;
	extern __shared__ unsigned char sharedMasks[];
	
	unsigned int cellStart = gridCellStarts[objectId];
	if (!cellStart)
		return;
	cellStart &= 0x7fffffff;
	unsigned int cellMortonCodeId = cellIds[cellStart];
	//unsigned int currentCellAbsoluteId = cellMortonCodeId + (ldexp((double)1, 3 * (levels - 1)) - 1) / 7; // ( 8^(levels - 1) - 1 ) / 7
	unsigned int currentLevel = levels;
	//printf("Thread %d in leaf Cells Count\n", objectId);
	unsigned int shift = ldexp((double)1, 3 * (levels - 1)) + 1;
	while (currentLevel > 0) {
		//unsigned int levelMask = MORTON_POSITION_MASK << (levels - currentLevel)*3;

		/*if (objectId % unsigned int(ldexp((double)1, 3 * (levels - currentLevel)))) {
			return;
		}*/

		unsigned int parentId = cellMortonCodeId >> 3; //(cellMortonCodeId & ~(levelMask)) >> (levels - currentLevel + 1) * 3;
		unsigned char childFillMask = 1 << unsigned int((cellMortonCodeId & MORTON_POSITION_MASK));
		unsigned int realParentId = shift / 8 + parentId;
		//printf("[%d] realParent %d pshift %d, parent %d, code %d, childMask %s, level %d\n", objectId, realParentId, shift / 8, parentId, cellMortonCodeId,/* int_bin_print(cellMortonCodeId),*/ char_bin_print(childFillMask), currentLevel);

		//printf("[%d] realParent %d shift %d, parent %d, code %d, morton %s, childMask %s, level %d\n", objectId, realParentId, shift, parentId, cellMortonCodeId, int_bin_print(cellMortonCodeId), char_bin_print(childFillMask), currentLevel);
		
		//if (!(masks[realParentId] & childFillMask)) {
		//atomicOr(masks + realParentId, childFillMask); //masks[parentId] |= childFillMask;
		//}

		/*sharedMasks[parentId] |= childFillMask;
		__syncthreads();

		if (objectId < ldexp((double)1, 3*(currentLevel - 1)) && !(objectId%4)) {
			unsigned char* memPtr = &masks[objectId];
			if (currentLevel > 1) {
				atomicOr((unsigned int*)(memPtr), *(unsigned int*)(&sharedMasks[parentId]));
				*(unsigned int*)(&sharedMasks[parentId]) = 0;
			}
			else {
				atomicOr((unsigned int*)(memPtr), (unsigned int)sharedMasks[parentId]);
			}
		}

		__syncthreads();*/
		//if (objectId < ldexp((double)1, 3 * (currentLevel - 1))) {
		if (!(masks[realParentId] & childFillMask)) {
			unsigned int* memPtr = (unsigned int*)masks + realParentId / 4;
			atomicOr(memPtr, (unsigned int)childFillMask << (8 * (realParentId % 4)));
			//printf("[%d] really writed to %d realId %d\n", objectId, parentId, realParentId);
		}
		else {
			return;
		}
		__syncthreads();
		/*atomicOr(masks + realParentId, childFillMask);
		__syncthreads();*/
		//}

		cellMortonCodeId = parentId;
		currentLevel--;
		shift = shift / 8;
	}
}



//void createOctreeGridData(const float* positionX, const float* positionY, const float* positionZ, unsigned int* gridCellIds,
//	unsigned int* particleIds, float cellWidth, float cellHeight, float cellDepth, unsigned int* treeData, unsigned char* masks,
//	unsigned int* shifts, unsigned int levels, unsigned int objectCount, unsigned int leafLayerCount)
//{
//	// Calculate launch parameters
//
//	const int threadsPerBlock = objectCount > 1024 ? 1024 : objectCount;
//	const int blocks = (objectCount + threadsPerBlock - 1) / threadsPerBlock;
//
//	// 1. Calculate cell id for every particle and store as pair (cell id, particle id) in two buffers
//	calculateOctreeCellIdKernel << <blocks, threadsPerBlock >> >
//		(positionX, positionY, positionZ, gridCellIds, particleIds, objectCount, cellWidth, cellHeight, cellDepth);
//	HANDLE_ERROR(cudaDeviceSynchronize());
//	
//
//	// 2. Sort particle ids by cell id
//
//	thrust::device_ptr<unsigned int> keys = thrust::device_pointer_cast<unsigned int>(gridCellIds);
//	thrust::device_ptr<unsigned int> values = thrust::device_pointer_cast<unsigned int>(particleIds);
//
//	thrust::stable_sort_by_key(keys, keys + objectCount, values);
//	
//	// 3. Find the start of every cell
//	const int threadsLastKernel = leafLayerCount > 1024 ? 1024 : leafLayerCount;
//	const int blocksLastKernel = (leafLayerCount + threadsLastKernel - 1) / threadsLastKernel;
//	calculateCellStarts <<<blocksLastKernel, threadsLastKernel >>> (gridCellIds, particleIds, treeData, objectCount);
//	HANDLE_ERROR(cudaPeekAtLastError());
//	printf("Keys: grid cell Ids, values: particle Ids\n");
//	printGpuKeyValueArray(gridCellIds, particleIds, objectCount);
//	//printf("-----------------------------------------------------------------\n");
//	////printGpuKeyValueArray(gridCellIds, treeData, objectCount);
//	printf("Pointers from leafs to starts of given grid cells:\n");
//	printGpuArray(treeData, leafLayerCount);
//	printf("masks in octree:\n");
//	size_t masksCount = 
//	HANDLE_ERROR(cudaMemset((void*)masks, 0, masksCount));
//	calculateTreeLeafsCells << <blocksLastKernel, threadsLastKernel, (threadsLastKernel/sizeof(unsigned int) + 1)*sizeof(unsigned char) >> > (gridCellIds, treeData, objectCount, levels, masks, shifts);
//	HANDLE_ERROR(cudaDeviceSynchronize());
//	HANDLE_ERROR(cudaPeekAtLastError());
//	printGpuArray(masks, (pow(8, levels + 1) - 1) / 7 - leafLayerCount);
//}

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

	cellCountX = cellCountY = cellCountZ = pow(2, levels);
	cellAmount = cellCountX * cellCountY * cellCountZ;

	HANDLE_ERROR(cudaMalloc((void**)&gridCellIds, objectsCount * sizeof(unsigned int)));
	HANDLE_ERROR(cudaMalloc((void**)&particleIds, objectsCount * sizeof(unsigned int)));
	HANDLE_ERROR(cudaMalloc((void**)&shifts, levels * sizeof(unsigned int)));

	unsigned int* shiftsHost = new unsigned int[levels];
	for(int i = 0, int index = 0; i < levels; ++i, index = index*8 + 1) {
		shiftsHost[i] = index;
	}
	HANDLE_ERROR(cudaMemcpy(shifts, shiftsHost,  levels*sizeof(unsigned int), cudaMemcpyHostToDevice));
	//HANDLE_ERROR(cudaMalloc((void**)&gridCellStarts, cellAmount * sizeof(unsigned int)));
	//HANDLE_ERROR(cudaMalloc((void**)&gridCellEnds, cellAmount * sizeof(unsigned int)));

	treeNodesCount = (pow(8, levels + 1) - 1) / 7;
	leafLayerCount = pow(8, levels );
	//printf("Octree nodes count: %d\n", treeNodesCount);
	size_t masksCount = sizeof(unsigned int) * ((treeNodesCount - leafLayerCount) / sizeof(unsigned int) + 1) * sizeof(unsigned char);
	HANDLE_ERROR(cudaMalloc((void**)&masks, masksCount*sizeof(unsigned char)));
	HANDLE_ERROR(cudaMalloc((void**)&treeData, leafLayerCount * sizeof(unsigned int)));
	HANDLE_ERROR(cudaMemset((void*)treeData, 0, leafLayerCount));
	HANDLE_ERROR(cudaPeekAtLastError());
}

void OctreeGrid::calculateGrid(const float* positionX, const float* positionY, const float* positionZ, unsigned int particleCount)
{
	/*createOctreeGridData(positionX, positionY, positionZ, gridCellIds, particleIds, cellWidth,
		cellHeight, cellDepth, treeData, masks, shifts, levels, particleCount, leafLayerCount);
	if (debugFrames < 2)
	{

	}
	if (!done && debugFrames++ == 2) {
		printFirstBytesFromGpu(masks, 585);
		done = true;
	}*/
	// Calculate launch parameters

	const int threadsPerBlock = particleCount > 1024 ? 1024 : particleCount;
	const int blocks = (particleCount + threadsPerBlock - 1) / threadsPerBlock;

	// 1. Calculate cell id for every particle and store as pair (cell id, particle id) in two buffers
	calculateOctreeCellIdKernel << <blocks, threadsPerBlock >> >
		(positionX, positionY, positionZ, gridCellIds, particleIds, particleCount, cellWidth, cellHeight, cellDepth);
	HANDLE_ERROR(cudaDeviceSynchronize());


	// 2. Sort particle ids by cell id

	thrust::device_ptr<unsigned int> keys = thrust::device_pointer_cast<unsigned int>(gridCellIds);
	thrust::device_ptr<unsigned int> values = thrust::device_pointer_cast<unsigned int>(particleIds);

	thrust::stable_sort_by_key(keys, keys + particleCount, values);

	// 3. Find the start of every cell
	const int threadsLastKernel = leafLayerCount > 1024 ? 1024 : leafLayerCount;
	const int blocksLastKernel = (leafLayerCount + threadsLastKernel - 1) / threadsLastKernel;
	calculateCellStarts << <blocksLastKernel, threadsLastKernel >> > (gridCellIds, particleIds, treeData, particleCount);
	HANDLE_ERROR(cudaPeekAtLastError());
	//printf("Keys: grid cell Ids, values: particle Ids\n");
	//printGpuKeyValueArray(gridCellIds, particleIds, particleCount);
	//printf("-----------------------------------------------------------------\n");
	////printGpuKeyValueArray(gridCellIds, treeData, particleCount);
	//("Pointers from leafs to starts of given grid cells:\n");
	//printGpuArray(treeData, leafLayerCount);
	//printf("masks in octree:\n");
	size_t masksCount = sizeof(unsigned int) * ((treeNodesCount - leafLayerCount) / sizeof(unsigned int) + 1) * sizeof(unsigned char);
	HANDLE_ERROR(cudaMemset((void*)masks, 0, masksCount));
	calculateTreeLeafsCells << <blocksLastKernel, threadsLastKernel, (threadsLastKernel / sizeof(unsigned int) + 1) * sizeof(unsigned char) >> > (gridCellIds, treeData, particleCount, levels, masks, shifts);
	HANDLE_ERROR(cudaDeviceSynchronize());
	HANDLE_ERROR(cudaPeekAtLastError());
	//printGpuArray(masks, (pow(8, levels + 1) - 1) / 7 - leafLayerCount);
}