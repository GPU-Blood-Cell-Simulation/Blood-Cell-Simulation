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
		if (length_squared(p1 - p2) <= 25.0f)
		{
			// Uncoalesced writes - area for optimization
			bloodCells.particles.force.set(particleId, 50.0f * normalize(p1 - p2));
		}
	}

	// Detect Colllisions in all 27 cells unless some corner cases are present - specified by template parameters. 
	template<int xMin, int xMax, int yMin, int yMax, int zMin, int zMax> 
	__device__ void detectCollisionsInNeighborCells(BloodCells& bloodCells, UniformGrid& grid, float3 p1, unsigned int particleId, unsigned int cellId)
	{
		#pragma unroll
		for (int x = xMin; x <= xMax; x++)
		{
			#pragma unroll
			for (int y = yMin; y <= yMax; y++)
			{
				#pragma unroll
				for (int z = zMin; z <= zMax; z++)
				{
					int neighborCellId = cellId + z * grid.cellCountX * grid.cellCountY + y * grid.cellCountX + x;

					for (int i = grid.gridCellStarts[neighborCellId]; i <= grid.gridCellEnds[neighborCellId]; i++)
					{
						int secondParticleId = grid.particleIds[i];

						// TODO: Potential optimization - unroll the loops manually or think of a way to metaprogram the compiler to unroll
						// one particular iteration (0,0,0) differently than the others
						if (particleId == secondParticleId)
							continue;

						float3 p2 = bloodCells.particles.position.get(secondParticleId);

						detectCollision(bloodCells, p1, p2, particleId);
					}
				}
			}
		}
	}


	// Should have been a deleted function but CUDA doesn't like it
	template<typename T>
	__global__ void calculateParticleCollisions(BloodCells bloodCells, T grid) {}


	// Calculate collisions between particles using UniformGrid
	template<>
	__global__ void calculateParticleCollisions<UniformGrid>(BloodCells bloodCells, UniformGrid grid)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= bloodCells.particleCount)
			return;

		int particleId = grid.particleIds[id];
		float3 p1 = bloodCells.particles.position.get(particleId);

		int cellId = grid.gridCellIds[id];
		int xId = static_cast<unsigned int>(bloodCells.particles.position.x[particleId] / grid.cellWidth);
		int yId = static_cast<unsigned int>(bloodCells.particles.position.y[particleId] / grid.cellHeight);
		int zId = static_cast<unsigned int>(bloodCells.particles.position.z[particleId] / grid.cellDepth);

		// Check all corner cases and call the appropriate function specialization
		// Ugly but fast
		if (xId < 1)
		{
			if (yId < 1)
			{
				if (zId < 1)
				{
					detectCollisionsInNeighborCells<0, 1, 0, 1, 0, 1>(bloodCells, grid, p1, particleId, cellId);
				}
				else if (zId > grid.cellCountZ - 2)
				{
					detectCollisionsInNeighborCells<0, 1, 0, 1, -1, 0>(bloodCells, grid, p1, particleId, cellId);
				}
				else
				{
					detectCollisionsInNeighborCells<0, 1, 0, 1, -1, 1>(bloodCells, grid, p1, particleId, cellId);
				}
			}
			else if (yId > grid.cellCountY - 2)
			{
				if (zId < 1)
				{
					detectCollisionsInNeighborCells<0, 1, -1, 0, 0, 1>(bloodCells, grid, p1, particleId, cellId);
				}
				else if (zId > grid.cellCountZ - 2)
				{
					detectCollisionsInNeighborCells<0, 1, -1, 0, -1, 0>(bloodCells, grid, p1, particleId, cellId);
				}
				else
				{
					detectCollisionsInNeighborCells<0, 1, -1, 0, -1, 1>(bloodCells, grid, p1, particleId, cellId);
				}
			}
			else
			{
				if (zId < 1)
				{
					detectCollisionsInNeighborCells<0, 1, -1, 1, 0, 1>(bloodCells, grid, p1, particleId, cellId);
				}
				else if (zId > grid.cellCountZ - 2)
				{
					detectCollisionsInNeighborCells<0, 1, -1, 1, -1, 0>(bloodCells, grid, p1, particleId, cellId);
				}
				else
				{
					detectCollisionsInNeighborCells<0, 1, -1, 1, -1, 1>(bloodCells, grid, p1, particleId, cellId);
				}
			}
		}
		else if (xId > grid.cellCountX - 2)
		{
			if (yId < 1)
			{
				if (zId < 1)
				{
					detectCollisionsInNeighborCells<-1, 0, 0, 1, 0, 1>(bloodCells, grid, p1, particleId, cellId);
				}
				else if (zId > grid.cellCountZ - 2)
				{
					detectCollisionsInNeighborCells<-1, 0, 0, 1, -1, 0>(bloodCells, grid, p1, particleId, cellId);
				}
				else
				{
					detectCollisionsInNeighborCells<-1, 0, 0, 1, -1, 1>(bloodCells, grid, p1, particleId, cellId);
				}
			}
			else if (yId > grid.cellCountY - 2)
			{
				if (zId < 1)
				{
					detectCollisionsInNeighborCells<-1, 0, -1, 0, 0, 1>(bloodCells, grid, p1, particleId, cellId);
				}
				else if (zId > grid.cellCountZ - 2)
				{
					detectCollisionsInNeighborCells<-1, 0, -1, 0, -1, 0>(bloodCells, grid, p1, particleId, cellId);
				}
				else
				{
					detectCollisionsInNeighborCells<-1, 0, -1, 0, -1, 1>(bloodCells, grid, p1, particleId, cellId);
				}
			}
			else
			{
				if (zId < 1)
				{
					detectCollisionsInNeighborCells<-1, 0, -1, 1, 0, 1>(bloodCells, grid, p1, particleId, cellId);
				}
				else if (zId > grid.cellCountZ - 2)
				{
					detectCollisionsInNeighborCells<-1, 0, -1, 1, -1, 0>(bloodCells, grid, p1, particleId, cellId);
				}
				else
				{
					detectCollisionsInNeighborCells<-1, 0, -1, 1, -1, 1>(bloodCells, grid, p1, particleId, cellId);
				}
			}
		}
		else
		{
			if (yId < 1)
			{
				if (zId < 1)
				{
					detectCollisionsInNeighborCells<-1, 1, 0, 1, 0, 1>(bloodCells, grid, p1, particleId, cellId);
				}
				else if (zId > grid.cellCountZ - 2)
				{
					detectCollisionsInNeighborCells<-1, 1, 0, 1, -1, 0>(bloodCells, grid, p1, particleId, cellId);
				}
				else
				{
					detectCollisionsInNeighborCells<-1, 1, 0, 1, -1, 1>(bloodCells, grid, p1, particleId, cellId);
				}
			}
			else if (yId > grid.cellCountY - 2)
			{
				if (zId < 1)
				{
					detectCollisionsInNeighborCells<-1, 1, -1, 0, 0, 1>(bloodCells, grid, p1, particleId, cellId);
				}
				else if (zId > grid.cellCountZ - 2)
				{
					detectCollisionsInNeighborCells<-1, 1, -1, 0, -1, 0>(bloodCells, grid, p1, particleId, cellId);
				}
				else
				{
					detectCollisionsInNeighborCells<-1, 1, -1, 0, -1, 1>(bloodCells, grid, p1, particleId, cellId);
				}
			}
			else
			{
				if (zId < 1)
				{
					detectCollisionsInNeighborCells<-1, 1, -1, 1, 0, 1>(bloodCells, grid, p1, particleId, cellId);
				}
				else if (zId > grid.cellCountZ - 2)
				{
					detectCollisionsInNeighborCells<-1, 1, -1, 1, -1, 0>(bloodCells, grid, p1, particleId, cellId);
				}
				else
				{
					detectCollisionsInNeighborCells<-1, 1, -1, 1, -1, 1>(bloodCells, grid, p1, particleId, cellId);
				}
			}
		}
	}

	// Calculate collisions between particles without any grid (naive implementation)
	template<>
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