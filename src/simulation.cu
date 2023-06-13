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

	__global__ void generateInitialPositionsKernel(Particles particles, Corpuscles corpuscles, float3 dims, int particleCount);


	// Allocate GPU buffers for the position vectors
	void allocateMemory(UniformGrid& grid, const unsigned int particleCount)
	{
		HANDLE_ERROR(cudaMalloc((void**)&grid.cellIds, particleCount * sizeof(unsigned int)));
		HANDLE_ERROR(cudaMalloc((void**)&grid.particleIds, particleCount * sizeof(unsigned int)));

		HANDLE_ERROR(cudaMalloc((void**)&grid.cellStarts, width / cellWidth * height / cellHeight * depth / cellDepth * sizeof(unsigned int)));
		HANDLE_ERROR(cudaMalloc((void**)&grid.cellEnds, width / cellWidth * height / cellHeight * depth / cellDepth * sizeof(unsigned int)));
	}

	// initial position approach
	// arg_0 should be a^2 * arg_1
	// but it is no necessary
	void generateInitialPositionsInLayers(Particles particles, Corpuscles corspuscles, const int particleCount, const int layersCount)
	{
		int modelParticleCount = 2; /* cell model particles count, 2 for dipole */
		int corpusclesPerLayer = particleCount / layersCount / modelParticleCount;
		int layerDim = sqrt(corpusclesPerLayer); // assuming layer is a square

		//real_particles_count = layerDim * layerDim * layersCount;

		int threadsPerBlockDim = std::min(layerDim, 32);
		int blDim = std::ceil(float(corpusclesPerLayer) / threadsPerBlockDim);
		dim3 blocks = dim3(blDim, blDim, layersCount);
		dim3 threads = dim3(threadsPerBlockDim, threadsPerBlockDim, 1);

		generateInitialPositionsKernel << <blocks, threads >> > (particles, corspuscles,
			make_float3(width, height, float(layersCount * depth) / 100), particleCount);
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


	__global__ void generateInitialPositionsKernel(Particles particles, Corpuscles corpuscles, float3 dims, int particleCount)
	{
		int thCnt = blockDim.x * blockDim.y;
		int blCnt2d = gridDim.x * gridDim.y;

		int tid = blockIdx.z * blCnt2d * thCnt + blockIdx.y * gridDim.x * thCnt
			+ blockIdx.x * thCnt + threadIdx.y * blockDim.x + threadIdx.x;

		float x = float((blockIdx.x * blockDim.x + threadIdx.x) * dims.x) / (blockDim.x * gridDim.x);
		float y = float((blockIdx.y * blockDim.y + threadIdx.y) * dims.y) / (blockDim.y * gridDim.y);
		float z = float((blockIdx.z * blockDim.z + threadIdx.z) * dims.z * blockDim.z) / (blockDim.z * gridDim.z * 100);

		//printf("id: %d, x: %f, y: %f, z: %f\n", tid, x, y, z);
		if (x <= dims.x && y <= dims.y)
		{
			corpuscles.setCorpuscle(tid, make_float3(x, y, z), particles, particleCount);
		}
	}


	__global__ void detectCollisions(Particles particles, Corpuscles corpuscls, unsigned int* cellIds, unsigned int* particleIds,
		unsigned int* cellStarts, unsigned int* cellEnds, unsigned int particleCount)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= particleCount)
			return;

		int particleId = particleIds[id];
		float3 p1 = particles.position.get(particleId);


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

			float3 p2 = particles.position.get(secondParticleId);
			if (length(p1 - p2) <= 5.0f)
			{
				// Uncoalesced writes - area for optimization
				particles.force.set(particleId, 50.0f * normalize(p1 - p2));
			}
		}
	}


	void calculateNextFrame(Particles particles, Corpuscles corpuscles, Triangles triangles, UniformGrid& grid, unsigned int particleCount)
	{
		// 1. calculate grid
		grid.calculateGrid(particles);

		int threadsPerBlock = particleCount > 1024 ? 1024 : particleCount;
		int blDim = std::ceil(float(particleCount) / threadsPerBlock);
		// 2. TODO: detections

		detectCollisions << < dim3(blDim), threadsPerBlock >> > (particles, corpuscles, grid.cellIds, grid.particleIds,
			grid.cellStarts, grid.cellEnds, particleCount);

		physics::propagateParticles << < dim3(blDim), threadsPerBlock >> > (particles, corpuscles, triangles, PARTICLE_COUNT, triangles.size);
	}
}