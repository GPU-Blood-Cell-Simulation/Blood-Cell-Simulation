#pragma once

#include "../blood_cell_structures/particles.cuh"
#include "base_grid.cuh"

/// <summary>
/// Represents a grid of uniform cell size
/// </summary>
class UniformGrid : public BaseGrid<UniformGrid>
{
private:
	bool isCopy = false;

public:
	unsigned int cellAmount;
	unsigned int* gridCellIds = 0;
	unsigned int* particleIds = 0;
	unsigned int* gridCellStarts = 0;
	unsigned int* gridCellEnds = 0;

	__host__ __device__ UniformGrid(const unsigned int particleCount);
	__host__ __device__ UniformGrid(const UniformGrid& other);
	__host__ __device__ ~UniformGrid();

	inline void calculateGrid(const Particles& particles, unsigned int particleCount)
	{
		calculateGrid(particles.position.x, particles.position.y, particles.position.z, particleCount);
	}

	void calculateGrid(const float* positionX, const float* positionY, const float* positionZ, unsigned int particleCount);

	__device__ unsigned int calculateCellId(float3 position);
};
