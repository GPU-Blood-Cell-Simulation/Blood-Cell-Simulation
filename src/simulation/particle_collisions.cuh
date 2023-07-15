#pragma once

#include "../grids/uniform_grid.cuh"
#include "../grids/no_grid.cuh"
#include "../blood_cell_structures/blood_cells.cuh"
#include "../utilities/math.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


namespace sim
{
	__device__ inline void detectCollision(BloodCells& bloodCells, float3 p1, float3 p2, unsigned int particleId)
	{
		if (length(p1 - p2) <= 5.0f)
		{
			// Uncoalesced writes - area for optimization
			bloodCells.particles.force.set(particleId, 50.0f * normalize(p1 - p2));
		}
	}


	template<typename T>
	__global__ void calculateParticleCollisions(BloodCells bloodCells, T grid) {}


	template<>
	// Calculate collisions between particles using UniformGrid
	__global__ void calculateParticleCollisions<UniformGrid>(BloodCells bloodCells, UniformGrid grid)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= bloodCells.particleCount)
			return;

		int particleId = grid.particleIds[id];
		float3 p1 = bloodCells.particles.position.get(particleId);

		int cellId = grid.gridCellIds[id];

		for (int i = grid.gridCellStarts[cellId]; i <= grid.gridCellEnds[cellId]; i++)
		{
			int secondParticleId = grid.particleIds[i];
			if (particleId == secondParticleId)
				continue;

			float3 p2 = bloodCells.particles.position.get(secondParticleId);

			detectCollision(bloodCells, p1, p2, particleId);
		}
	}

	template<>
	// Calculate collisions between particles without any grid (naive implementation)
	__global__ void calculateParticleCollisions<NoGrid>(BloodCells bloodCells, NoGrid grid)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= bloodCells.particleCount)
			return;

		float3 p1 = bloodCells.particles.position.get(id);

		// Naive implementation
		for (int i = 0; i < bloodCells.particleCount; i++)
		{
			if (id == i)
				continue;

			float3 p2 = bloodCells.particles.position.get(i);

			detectCollision(bloodCells, p1, p2, id);
		}
	}
}