#pragma once

#include "../objects/particles.cuh"
#include "base_grid.cuh"
#include "../defines.hpp"

/// <summary>
/// Represents a grid of uniform cell size
/// </summary>
class UniformGrid : public BaseGrid<UniformGrid>
{
private:
	bool isCopy = false;

public:

	int cellWidth;
	int cellHeight;
	int cellDepth;
	int cellCountX;
	int cellCountY;
	int cellCountZ;
	int cellAmount;
	int objectCount;

	int* gridCellIds = 0;
	int* particleIds = 0;
	int* gridCellStarts = 0;
	int* gridCellEnds = 0;

	UniformGrid(const int objectCount, int cellWidth, int cellHeight, int cellDepth);
	UniformGrid(const UniformGrid& other);
	~UniformGrid();

	inline void calculateGrid(const Particles& particles, int objectCount,const cudaStream_t& stream)
	{
		calculateGrid(particles.positions.x, particles.positions.y, particles.positions.z, objectCount, stream);
	}

	void calculateGrid(const float* positionX, const float* positionY, const float* positionZ, int objectCount, const cudaStream_t& stream);

	__device__ int calculateCellId(float3 position);
};
