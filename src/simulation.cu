#include "simulation.cuh"
#include "grid.cuh"
#include "defines.cuh"
#include "physics.cuh"

#include <cmath>
#include <ctime>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace sim
{

	__global__ void setupCurandStatesKernel(curandState* states, const unsigned long seed, const int particleCount);

	__global__ void generateRandomPositionsKernel(curandState* states, Particles p, const int particleCount);

	__global__ void generateInitialPositionsKernel(BloodCells cells);

	// initial position approach
	// arg_0 should be a^2 * arg_1
	// but it is no necessary
	//void generateInitialPositionsInLayers(Particles p, dipols c, const int particleCount, const int layersCount)
	//{
	//	int model_par_cnt = 2; /* cell model particles count, 2 for dipole */
	//	int corpusclesPerLayer = particleCount / layersCount / model_par_cnt;
	//	int layerDim = sqrt(corpusclesPerLayer); // assuming layer is a square

	//	//real_particles_count = layerDim * layerDim * layersCount;

	//	int threadsPerBlockDim = std::min(layerDim, 32);
	//	int blDim = std::ceil(float(corpusclesPerLayer) / threadsPerBlockDim);
	//	dim3 blocks = dim3(blDim, blDim, layersCount);
	//	dim3 threads = dim3(threadsPerBlockDim, threadsPerBlockDim, 1);

	//	generateInitialPositionsKernel << <blocks, threads >> > (p, c,
	//		make_float3(width, height, float(layersCount * depth) / 100), particleCount);
	//}


	// Generate initial positions and velocities of particles
	void generateRandomPositions(Particles p, const int particleCount)
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

		generateRandomPositionsKernel << <blocks, threadsPerBlock >> > (devStates, p, particleCount);

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

		p.position.x[id] = curand_uniform(&states[id])* width;
		p.position.y[id] = curand_uniform(&states[id])* height;
		p.position.z[id] = curand_uniform(&states[id])* depth;

		p.force.x[id] = 0;
		p.force.y[id] = 0;
		p.force.z[id] = 0;
	}


/*
	__global__ void generateInitialPositionsKernel(Particles par, dipols crps, float3 dims, int par_cnt)
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
			crps.setCorpuscle(tid, make_float3(x, y, z), par, par_cnt);
		}
	}
*/

	__global__ void detectCollisions(BloodCells cells)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= cells.particlesCnt)
			return;

		float3 p1 = cells.particles.position.get(id);

		for (int i = 0; i < cells.particlesCnt; i++) {
			if (id == i)
				continue;

			float3 p2 = cells.particles.position.get(i);
			if (length(p1 - p2) <= 5.0f) {
				cells.particles.force.set(id, 10.0f * normalize(p1 - p2));
			}
		}
	}


	void calculateNextFrame(BloodCells cells)
	{
		int threadsPerBlock = cells.particlesCnt > 1024 ? 1024 : cells.particlesCnt;
		int blDim = std::ceil(float(cells.particlesCnt) / threadsPerBlock);
		
		detectCollisions << < dim3(blDim), threadsPerBlock >> > (cells);

		physics::propagateParticles << < dim3(blDim), threadsPerBlock >> > (cells);

		cells.PropagateForces();
	}
}
