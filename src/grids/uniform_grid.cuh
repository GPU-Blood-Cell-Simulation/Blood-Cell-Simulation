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
	unsigned int* cellAmountDevice;

	unsigned int* gridCellIds = 0;
	unsigned int* particleIds = 0;
	unsigned int* gridCellStarts = 0;
	unsigned int* gridCellEnds = 0;

	__host__ __device__ UniformGrid(const unsigned int particleCount, unsigned int cellWidth, unsigned int cellHeight, unsigned int cellDepth);
	__host__ __device__ UniformGrid(const UniformGrid& other);
	__host__ __device__ ~UniformGrid();

	inline void calculateGrid(const Particles& particles, unsigned int particleCount)
	{
		calculateGrid(particles.position.x, particles.position.y, particles.position.z, particleCount);
	}

	__device__ unsigned int calculateIdForCell(float x, float y, float z)
	{
		return
			static_cast<unsigned int>(z / cellDepth) * static_cast<unsigned int>(width / cellWidth) * static_cast<unsigned int>(height / cellHeight) +
			static_cast<unsigned int>(y / cellHeight) * static_cast<unsigned int>(width / cellWidth) +
			static_cast<unsigned int>(x / cellWidth);
	}

	__global__ void calculateCellIdKernel(const float* positionX, const float* positionY, const float* positionZ,
		unsigned int* cellIds, unsigned int* particleIds, const unsigned int particleCount)
	{
		unsigned int particleId = blockIdx.x * blockDim.x + threadIdx.x;
		if (particleId >= particleCount)
			return;
		/*unsigned int cellId =
			static_cast<unsigned int>(positionZ[particleId] / cellDepth) * static_cast<unsigned int>(width / cellWidth) * static_cast<unsigned int>(height / cellHeight) +
			static_cast<unsigned int>(positionY[particleId] / cellHeight) * static_cast<unsigned int>(width / cellWidth) +
			static_cast<unsigned int>(positionX[particleId] / cellWidth);*/
		unsigned int cellId = calculateIdForCell(positionX[particleId], positionY[particleId], positionZ[particleId]);
		// Debug
		/*if (cellId >= 9261)
			printf("Error, cellId: %d\n", cellId);*/

			//printf("id: %d, cellId: %d\n", particleId, cellId);

		particleIds[particleId] = particleId;
		cellIds[particleId] = cellId;

	}
	void calculateGrid(const float* positionX, const float* positionY, const float* positionZ, unsigned int particleCount);

	__device__ unsigned int calculateCellId(float3 position);
};
