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

	__global__ void generateRandomPositionsKernel(curandState* states, Particles particles, const int particleCount);

	SimulationController::SimulationController(BloodCells& bloodCells, DeviceTriangles& triangles, Grid grid) : bloodCells(bloodCells), triangles(triangles), grid(grid)
	{
		// Generate random particle positions
		generateRandomPositions();
	}

	SimulationController::~SimulationController()
	{

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

		generateRandomPositionsKernel << <blocks, threadsPerBlock >> > (devStates, bloodCells.particles, bloodCells.particleCount);

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
	__global__ void generateRandomPositionsKernel(curandState* states, Particles p, const int particleCount)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= particleCount)
			return;

		p.position.x[id] = curand_uniform(&states[id]) * width / 3 + width/6;
		p.position.y[id] = curand_uniform(&states[id]) * height / 3 + height/6;
		p.position.z[id] = curand_uniform(&states[id]) * depth / 3 + depth/6;

		p.force.x[id] = 0;
		p.force.y[id] = 0;
		p.force.z[id] = 0;
	}

	// Main simulation function, called every frame
	void SimulationController::calculateNextFrame()
	{
		// 1. Calculate grid
		std::visit([&](auto&& g)
			{
				g->calculateGrid(bloodCells.particles, bloodCells.particleCount);
			}, grid);
		

		int threadsPerBlock = bloodCells.particleCount > 1024 ? 1024 : bloodCells.particleCount;
		int blDim = std::ceil(float(bloodCells.particleCount) / threadsPerBlock);
		
		// 2. Detect particle collisions
		std::visit([&](auto&& g)
			{
				calculateParticleCollisions << < dim3(blDim), threadsPerBlock >> > (bloodCells, *g);
			}, grid);
		

		// 3. Propagate particle forces into neighbors

		bloodCells.propagateForces();

		// 4. Detect vein collisions and propagate forces -> velocities, velocities -> positions

		detectVeinCollisionsAndPropagateParticles << < dim3(blDim), threadsPerBlock >> > (bloodCells, triangles);

		// 5. Propagate forces -> velocities, velocities -> positions for vein triangles
		threadsPerBlock = triangles.vertexCount > 1024 ? 1024 : triangles.vertexCount;
		blDim = std::ceil(float(triangles.vertexCount) / threadsPerBlock);

		propagateVeinTriangleVertices << < dim3(blDim), threadsPerBlock >> > (triangles);
	}
}