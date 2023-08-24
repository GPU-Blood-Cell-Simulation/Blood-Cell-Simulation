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

	__global__ void generateRandomPositionsKernel(curandState* states, Particles particles, const int particleCount, glm::vec3 cylinderBaseCenter/*, float cylinderRadius, float cylinderHeight*/);

	SimulationController::SimulationController(BloodCells& bloodCells, DeviceTriangles& triangles, Grid grid) : bloodCells(bloodCells), triangles(triangles), grid(grid)
	{
		// Generate random particle positions
		generateRandomPositions();
	}

	// Generate initial positions and velocities of particles
	void SimulationController::generateRandomPositions()
	{
		int threadsPerBlock = bloodCells.particleCount > 1024 ? 1024 : bloodCells.particleCount;
		int blocks = (bloodCells.particleCount + threadsPerBlock - 1) / threadsPerBlock;

		// Set up random seeds
		curandState* devStates;
		HANDLE_ERROR(cudaMalloc(&devStates, bloodCells.particleCount * sizeof(curandState)));
		srand(static_cast<unsigned int>(time(0)));
		int seed = rand();
		setupCurandStatesKernel << <blocks, threadsPerBlock >> > (devStates, seed, bloodCells.particleCount);

		// Generate random positions and velocity vectors

		generateRandomPositionsKernel << <blocks, threadsPerBlock >> > (devStates, bloodCells.particles, bloodCells.particleCount, cylinderBaseCenter);


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
	__global__ void generateRandomPositionsKernel(curandState* states, Particles p, const int particleCount, glm::vec3 cylinderBaseCenter/*, float cylinderRadius, float cylinderHeight*/)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= particleCount)
			return;

		p.position.x[id] = cylinderBaseCenter.x - cylinderRadius * 0.5f + curand_uniform(&states[id]) * cylinderRadius;
		p.position.y[id] = cylinderBaseCenter.y - cylinderRadius * 0.5f + curand_uniform(&states[id]) * cylinderRadius + cylinderHeight/2;
		p.position.z[id] = cylinderBaseCenter.z - cylinderRadius * 0.5f + curand_uniform(&states[id]) * cylinderRadius;

		p.force.x[id] = 0;
		p.force.y[id] = 0;
		p.force.z[id] = 0;
	}

	// Main simulation function, called every frame
	void SimulationController::calculateNextFrame()
	{
    // 1. Calculate grids
		std::visit([&](auto&& g1, auto&& g2)
			{
				// 1. Calculate grids
				g1->calculateGrid(bloodCells.particles, bloodCells.particleCount);
				g2->calculateGrid(triangles.centers.x, triangles.centers.y, triangles.centers.z, triangleCount);

				// anything above 768 threads (25 warps) trigger an error
				// 'too many resources requested for launch'
				// maybe possible to solve
				int threadsPerBlock = bloodCells.particleCount > 768 ? 768 : bloodCells.particleCount;
				int blDim = std::ceil(float(bloodCells.particleCount) / threadsPerBlock);

				// 2. Detect particle collisions
				calculateParticleCollisions << < dim3(blDim), threadsPerBlock >> > (bloodCells, *g1);
				HANDLE_ERROR(cudaPeekAtLastError());

				// 3. Propagate particle forces into neighbors

        bloodCells.propagateForces();
        HANDLE_ERROR(cudaPeekAtLastError());
    
        // 4. Detect vein collisions and propagate forces -> velocities, velocities -> positions

        detectVeinCollisionsAndPropagateParticles << < dim3(blDim), threadsPerBlock >> > (bloodCells, triangles);
        HANDLE_ERROR(cudaPeekAtLastError());
    
        // 5. Gather forces from neighbors

        triangles.gatherForcesFromNeighbors();
        HANDLE_ERROR(cudaPeekAtLastError());
    
        // 6. Propagate forces -> velocities, velocities -> positions for vein triangles
        triangles.propagateForcesIntoPositions();
        HANDLE_ERROR(cudaPeekAtLastError());
    
				// 5. Recalculate triangles centers
				triangles.calculateCenters();
				HANDLE_ERROR(cudaPeekAtLastError());

			}, particleGrid, triangleGrid);
	}
}