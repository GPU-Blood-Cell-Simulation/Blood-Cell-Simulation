#include "simulation.cuh"
#include "../defines.hpp"
#include "particle_collisions.cuh"
#include "vein_collisions.cuh"
#include "../blood_cell_structures/device_triangles.cuh"
#include "../utilities/cuda_handle_error.cuh"

#include <cmath>
#include <ctime>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


namespace sim
{
	__global__ void setupCurandStatesKernel(curandState* states, const unsigned long seed, const int particleCount);

	__global__ void generateRandomPositionsKernel(curandState* states, Particles particles, const int particleCount);

	// Generate initial positions and velocities of particles
	void generateRandomPositions(Particles& particles, const int particleCount)
	{
		int threadsPerBlock = particleCount > 1024 ? 1024 : particleCount;
		int blocks = (particleCount + threadsPerBlock - 1) / threadsPerBlock;

		// Set up random seeds
		curandState* devStates;
		HANDLE_ERROR(cudaMalloc(&devStates, particleCount * sizeof(curandState)));
		srand(static_cast<unsigned int>(time(0)));
		int seed = rand();
		setupCurandStatesKernel << <blocks, threadsPerBlock >> > (devStates, seed, particleCount);

		// Generate random positions and velocity vectors

		generateRandomPositionsKernel << <blocks, threadsPerBlock >> > (devStates, particles, particleCount);

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
	void calculateNextFrame(BloodCells& cells, DeviceTriangles& triangles, UniformGrid& grid, unsigned int triangleCount)
	{
		// 1. Calculate grid
		grid.calculateGrid(cells.particles);

		int threadsPerBlock = cells.particleCount > 1024 ? 1024 : cells.particleCount;
		int blDim = std::ceil(float(cells.particleCount) / threadsPerBlock);
		
		// 2. Detect particle collisions

		detectParticleCollisions << < dim3(blDim), threadsPerBlock >> > (cells, grid.gridCellIds, grid.particleIds,
			grid.gridCellStarts, grid.gridCellEnds);

		// 3. Propagate forces into neighbors

		cells.PropagateForces();

		// 4. Detect vein collisions and propagate forces -> velocities, velocities -> positions

		detectVeinCollisionsAndPropagateParticles << < dim3(blDim), threadsPerBlock >> > (cells, triangles);
	}
}