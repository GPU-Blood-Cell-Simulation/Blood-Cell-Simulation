#pragma once
#include "base_grid.cuh"
#include "../blood_cell_structures/particles.cuh"

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
	unsigned int* gridCellStarts = 0;
	unsigned int* gridCellEnds = 0;

	OctreeGrid(const unsigned int particleCount, unsigned int cellWidth, unsigned int cellHeight, unsigned int cellDepth);
	OctreeGrid(const OctreeGrid& other);
	~OctreeGrid();

	inline void calculateGrid(const Particles& particles, unsigned int particleCount)
	{
		calculateGrid(particles.position.x, particles.position.y, particles.position.z, particleCount);
	}

	void calculateGrid(const float* positionX, const float* positionY, const float* positionZ, unsigned int particleCount) {}

	__device__ unsigned int calculateCellId(float3 position) {}
};
