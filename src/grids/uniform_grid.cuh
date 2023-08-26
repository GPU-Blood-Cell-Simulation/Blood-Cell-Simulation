#pragma once

#include "../objects/particles.cuh"
#include "base_grid.cuh"

/// <summary>
/// Represents a grid of uniform cell size
/// </summary>
class UniformGrid : public BaseGrid<UniformGrid>
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

	UniformGrid(const unsigned int particleCount, unsigned int cellWidth, unsigned int cellHeight, unsigned int cellDepth);
	UniformGrid(const UniformGrid& other);
	~UniformGrid();

	inline void calculateGrid(const Particles& particles, unsigned int particleCount)
	{
		calculateGrid(particles.positions.x, particles.positions.y, particles.positions.z, particleCount);
	}

	void calculateGrid(const float* positionX, const float* positionY, const float* positionZ, unsigned int particleCount);

	__device__ unsigned int calculateCellId(float3 position);
};
