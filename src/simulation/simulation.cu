#include "simulation.cuh"
#include "../defines.hpp"
#include "vein_collisions.cuh"
#include "particle_collisions.cuh"
#include "../utilities/cuda_handle_error.cuh"

#include <cmath>
#include <ctime>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


namespace sim
{
	__global__ void setupCurandStatesKernel(curandState* states, const unsigned long seed, const int particleCount);

	__global__ void generateRandomPositionsKernel(curandState* states, Particles particles, const unsigned int particleCount, glm::vec3 cylinderBaseCenter/*, float cylinderRadius, float cylinderHeight*/);


	// TODO :
	// anything above 768 threads (25 warps) trigger an error
	// 'too many resources requested for launch'
	// maybe possible to solve
	inline constexpr unsigned int maxThreads = 768;

	SimulationController::SimulationController(BloodCells& bloodCells, VeinTriangles& triangles, Grid particleGrid, Grid triangleGrid) :
		bloodCells(bloodCells), triangles(triangles), particleGrid(particleGrid), triangleGrid(triangleGrid),
		bloodCellsThreadsPerBlock(bloodCells.particleCount > maxThreads ? maxThreads : bloodCells.particleCount),
		bloodCellsBlocks(std::ceil(static_cast<float>(bloodCells.particleCount) / bloodCellsThreadsPerBlock)),
		veinVerticesThreadsPerBlock(triangles.vertexCount > maxThreads ? maxThreads : triangles.vertexCount),
		veinVerticesBlocks(std::ceil(static_cast<float>(triangles.vertexCount) / veinVerticesThreadsPerBlock)),
		veinTrianglesThreadsPerBlock(triangles.triangleCount > maxThreads ? maxThreads : triangles.triangleCount),
		veinTrianglesBlocks(std::ceil(static_cast<float>(triangles.triangleCount) / veinTrianglesThreadsPerBlock))
	{
		// Generate random particle positions
		generateRandomPositions();
	}

	// Generate initial positions and velocities of particles
	void SimulationController::generateRandomPositions()
	{

		// Set up random seeds
		curandState* devStates;
		HANDLE_ERROR(cudaMalloc(&devStates, bloodCells.particleCount * sizeof(curandState)));
		srand(static_cast<unsigned int>(time(0)));
		int seed = rand();
		setupCurandStatesKernel << <bloodCellsBlocks, bloodCellsThreadsPerBlock >> > (devStates, seed, bloodCells.particleCount);

		// Generate random positions and velocity vectors

		generateRandomPositionsKernel << <bloodCellsBlocks, bloodCellsThreadsPerBlock >> > (devStates, bloodCells.particles, bloodCells.particleCount, cylinderBaseCenter);


		HANDLE_ERROR(cudaFree(devStates));
	}

	__global__ void setupCurandStatesKernel(curandState* states, const unsigned long seed, const int particleCount)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= particleCount)
			return;
		curand_init(seed, id, 0, &states[id]);
	}

	// Generate random positions and velocities at the beginning
	__global__ void generateRandomPositionsKernel(curandState* states, Particles particles, const unsigned int particleCount, glm::vec3 cylinderBaseCenter/*, float cylinderRadius, float cylinderHeight*/)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= particleCount)
			return;

		particles.position.x[id] = cylinderBaseCenter.x - cylinderRadius * 0.5f + curand_uniform(&states[id]) * cylinderRadius;
		particles.position.y[id] = cylinderBaseCenter.y - cylinderRadius * 0.5f + curand_uniform(&states[id]) * cylinderRadius + cylinderHeight/2;
		particles.position.z[id] = cylinderBaseCenter.z - cylinderRadius * 0.5f + curand_uniform(&states[id]) * cylinderRadius;

		particles.velocity.x[id] = 0;
		particles.velocity.y[id] = 0;
		particles.velocity.z[id] = 0;

		particles.force.x[id] = 0;
		particles.force.y[id] = 0;
		particles.force.z[id] = 0;
	}

	// Main simulation function, called every frame
	void SimulationController::calculateNextFrame()
	{
		std::visit([&](auto&& g1, auto&& g2)
			{
				// 1. Calculate grids
				g1->calculateGrid(bloodCells.particles, bloodCells.particleCount);
				g2->calculateGrid(triangles.centers.x, triangles.centers.y, triangles.centers.z, triangles.triangleCount);

				// 2. Detect particle collisions
				calculateParticleCollisions << < bloodCellsBlocks, bloodCellsThreadsPerBlock >> > (bloodCells, *g1);
				HANDLE_ERROR(cudaPeekAtLastError());

				// 3. Propagate particle forces into neighbors

				bloodCells.gatherForcesFromNeighbors(bloodCellsBlocks, bloodCellsThreadsPerBlock);
				HANDLE_ERROR(cudaPeekAtLastError());
    
				// 4. Detect vein collisions and propagate forces -> velocities, velocities -> positions

				detectVeinCollisionsAndPropagateParticles << < bloodCellsBlocks, bloodCellsThreadsPerBlock >> > (bloodCells, triangles, *g1, *g2);
				HANDLE_ERROR(cudaPeekAtLastError());
    
				// 5. Gather forces from neighbors

				triangles.gatherForcesFromNeighbors(veinVerticesBlocks, veinVerticesThreadsPerBlock);
				HANDLE_ERROR(cudaPeekAtLastError());
    
				// 6. Propagate forces -> velocities, velocities -> positions for vein triangles
				triangles.propagateForcesIntoPositions(veinVerticesBlocks, veinVerticesThreadsPerBlock);
				HANDLE_ERROR(cudaPeekAtLastError());
    
				// 7. Recalculate triangles centers
				triangles.calculateCenters(veinTrianglesBlocks, veinTrianglesThreadsPerBlock);
				HANDLE_ERROR(cudaPeekAtLastError());

			}, particleGrid, triangleGrid);
	}
}