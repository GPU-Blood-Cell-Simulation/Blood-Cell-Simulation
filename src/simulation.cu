#include "simulation.cuh"
#include "defines.cuh"
#include "physics.cuh"

#include <cmath>
#include <ctime>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


namespace sim
{

	__global__ void setupCurandStatesKernel(curandState* states, const unsigned long seed, const int particleCount);

	__global__ void generateRandomPositionsKernel(curandState* states, Particles particles, const int particleCount);


	// Allocate GPU buffers for the position vectors
	void allocateMemory(UniformGrid& grid, const unsigned int particleCount)
	{
		HANDLE_ERROR(cudaMalloc((void**)&grid.cellIds, particleCount * sizeof(unsigned int)));
		HANDLE_ERROR(cudaMalloc((void**)&grid.particleIds, particleCount * sizeof(unsigned int)));

		HANDLE_ERROR(cudaMalloc((void**)&grid.cellStarts, width / cellWidth * height / cellHeight * depth / cellDepth * sizeof(unsigned int)));
		HANDLE_ERROR(cudaMalloc((void**)&grid.cellEnds, width / cellWidth * height / cellHeight * depth / cellDepth * sizeof(unsigned int)));
	}


	// Generate initial positions and velocities of particles
	void generateRandomPositions(Particles particles, const int particleCount)
	{
		int threadsPerBlock = particleCount > 1024 ? 1024 : particleCount;
		int blocks = (particleCount + threadsPerBlock - 1) / threadsPerBlock;

		// Set up random seeds
		curandState* devStates;
		cudaMalloc(&devStates, particleCount * sizeof(curandState));
		srand(static_cast<unsigned int>(time(0)));
		int seed = rand();
		setupCurandStatesKernel << <blocks, threadsPerBlock >> > (devStates, seed, particleCount);

		// Generate random positions and velocity vectors

		generateRandomPositionsKernel << <blocks, threadsPerBlock >> > (devStates, particles, particleCount);

		cudaFree(devStates);
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


	__global__ void detectCollisions(BloodCells cells, unsigned int* cellIds, unsigned int* particleIds,
		unsigned int* cellStarts, unsigned int* cellEnds)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= cells.particlesCnt)
			return;

		int particleId = particleIds[id];
		float3 p1 = cells.particles.position.get(particleId);


		// Naive implementation
		/*for (int i = 0; i < particleCount; i++)
		{
			if (id == i || i == secondParticle)
				continue;

			float3 p2 = particles.position.get(i);
			if (length(p1 - p2) <= 5.0f)
			{
				particles.force.set(id, 50.0f * normalize(p1 - p2));
			}
		}*/

		// Using uniform grid

		int cellId = cellIds[id];

		for (int i = cellStarts[cellId]; i <= cellEnds[cellId]; i++)
		{
			int secondParticleId = particleIds[i];
			if (particleId == secondParticleId)
				continue;

			float3 p2 = cells.particles.position.get(secondParticleId);
			if (length(p1 - p2) <= 5.0f)
			{
				// Uncoalesced writes - area for optimization
				cells.particles.force.set(particleId, 50.0f * normalize(p1 - p2));
			}
		}
	}

	void calculateNextFrame(BloodCells cells, DeviceTriangles triangles, UniformGrid& grid, unsigned int trianglesCount)
	{
		// 1. calculate grid
		grid.calculateGrid(cells.particles);

		int threadsPerBlock = cells.particlesCnt > 1024 ? 1024 : cells.particlesCnt;
		int blDim = std::ceil(float(cells.particlesCnt) / threadsPerBlock);
		
		// 2. TODO: detections

		detectCollisions << < dim3(blDim), threadsPerBlock >> > (cells, grid.cellIds, grid.particleIds,
			grid.cellStarts, grid.cellEnds);


		physics::propagateParticles << < dim3(blDim), threadsPerBlock >> > (cells, triangles, trianglesCount);
		cells.PropagateForces();
	}
}