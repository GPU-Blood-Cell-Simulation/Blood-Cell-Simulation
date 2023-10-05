#pragma once
#include "base_grid.cuh"
#include "../objects/particles.cuh"
#include <vector>

/// <summary>
/// Represents an implementation of a Sparse voxel octree
/// </summary>
class OctreeGrid : public BaseGrid<OctreeGrid>
{
private:
	bool isCopy = false;

public:

	unsigned int cellWidth;
	unsigned int cellHeight;
	unsigned int cellDepth;
	unsigned int cellCountX;
	unsigned int cellCountY;
	unsigned int cellCountZ;
	unsigned int cellAmount;
	unsigned int objectsCount;

	unsigned int* gridCellIds = 0;
	unsigned int* particleIds = 0;
	unsigned int* shifts;
	/*unsigned int* gridCellStarts = 0;
	unsigned int* gridCellEnds = 0;*/

	unsigned int levels;
	unsigned int treeNodesCount;
	unsigned int leafLayerCount;
	
	unsigned int* treeData;
	unsigned char* masks;

	OctreeGrid(const unsigned int objectCount, const unsigned int levels);
	OctreeGrid(const OctreeGrid& other);
	~OctreeGrid();

	inline void calculateGrid(const Particles& particles, unsigned int particleCount)
	{
		calculateGrid(particles.positions.x, particles.positions.y, particles.positions.z, particleCount);
	}

	void calculateGrid(const float* positionX, const float* positionY, const float* positionZ, unsigned int particleCount);

	//__device__ void traverseGrid(float3 origin, float3 direction, float tmax);

	__device__ unsigned int calculateCellId(float3 position) {}
};

__device__ unsigned int partEveryByteByTwo(unsigned int n);

__device__ unsigned int calculateMortonCodeIdForCell(unsigned int xId, unsigned int yId, unsigned int zId);

__global__ void calculateOctreeCellIdKernel(const float* positionX, const float* positionY, const float* positionZ,
	unsigned int* cellIds, unsigned int* particleIds, const unsigned int particleCount,
	unsigned int cellWidth, unsigned int cellHeight, unsigned int cellDepth);

__global__ void calculateCellStarts(const unsigned int* cellIds, const unsigned int* particleIds,
	unsigned int* treeData, unsigned int cellCount);
 
__global__ void calculateTreeLeafsCells(const unsigned int* cellIds, const unsigned int objectCount, const unsigned int levels
	, unsigned char* masks, unsigned int* shifts);

void createOctreeGridData(const float* positionX, const float* positionY, const float* positionZ, unsigned int* gridCellIds,
	unsigned int* particleIds, float cellWidth, float cellHeight, float cellDepth, unsigned int* treeData, unsigned char* masks,
	unsigned int* shifts, unsigned int levels, unsigned int objectCount);