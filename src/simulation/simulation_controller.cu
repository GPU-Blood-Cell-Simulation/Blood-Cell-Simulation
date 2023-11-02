#include "simulation_controller.cuh"

#include "../meta_factory/blood_cell_factory.hpp"
#include "../meta_factory/vein_factory.hpp"
#include "../objects/particles.cuh"
#include "particle_collisions.cuh"
#include "../utilities/cuda_handle_error.cuh"
#include "vein_collisions.cuh"
#include "vein_end.cuh"

#include <cmath>
#include <ctime>
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>

namespace sim
{
	__global__ void setupCurandStatesKernel(curandState* states, unsigned long seed);

	__global__ void generateRandomPositionsKernel(curandState* states, Particles particles, glm::vec3 cylinderBaseCenter, 
		cudaVec3 bloodCellModelPosition, unsigned int bloodCellModelSize);


	SimulationController::SimulationController(BloodCells& bloodCells, VeinTriangles& triangles, std::vector<glm::vec3> bloodCellVertices, unsigned int bloodCellSize ,Grid particleGrid, Grid triangleGrid) :
		bloodCells(bloodCells), triangles(triangles), particleGrid(particleGrid), triangleGrid(triangleGrid),
		bloodCellsThreads(particleCount),
		veinVerticesThreads(triangles.vertexCount),
		veinTrianglesThreads(triangles.triangleCount),
		bloodCellSize(bloodCellSize),
		bloodCellModel(bloodCellSize)
	{
		// Create streams
		for (int i = 0; i < bloodCellTypeCount; i++)
		{
			streams[i] = cudaStream_t();
			HANDLE_ERROR(cudaStreamCreate(&streams[i]));
		}

		float* xmodel = new float[bloodCellSize];
		float* ymodel = new float[bloodCellSize];
		float* zmodel = new float[bloodCellSize];

		int i = 0;
		std::for_each(bloodCellVertices.begin(), bloodCellVertices.end(), [&](auto& v) {
			xmodel[i] = v.x[i];
			ymodel[i] = v.x[i];
			zmodel[i] = v.x[i];
		});

		HANDLE_ERROR(cudaMemcpy(bloodCellModel.x, xmodel, bloodCellSize * sizeof(float), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(bloodCellModel.y, ymodel, bloodCellSize * sizeof(float), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(bloodCellModel.z, zmodel, bloodCellSize * sizeof(float), cudaMemcpyHostToDevice));
		delete[] xmodel;
		delete[] ymodel;
		delete[] zmodel;
		// Generate random particle positions
		generateRandomPositions();
	}

	sim::SimulationController::~SimulationController()
	{
		for (int i = 0; i < bloodCellTypeCount; i++)
		{
			HANDLE_ERROR(cudaStreamDestroy(streams[i]));
		}
	}

	// Generate initial positions and velocities of particles
	void SimulationController::generateRandomPositions()
	{
		// Set up random seeds
		curandState* devStates;
		HANDLE_ERROR(cudaMalloc(&devStates, particleCount * sizeof(curandState)));
		srand(static_cast<unsigned int>(time(0)));
		int seed = rand();
		setupCurandStatesKernel << <bloodCellsThreads.blocks, bloodCellsThreads.threadsPerBlock >> > (devStates, seed);

		// Generate random positions and velocity vectors
		using IndexList = mp_iota_c<bloodCellTypeCount>;
		mp_for_each<IndexList>([&](auto i)
		{
			using BloodCellDefinition = mp_at_c<BloodCellList, i>;
			constexpr int particlesStart = particlesStarts[i];
			constexpr int graphStart = accumulatedGraphSizes[i];

			CudaThreads threads(BloodCellDefinition::count * BloodCellDefinition::particlesInCell);
			generateRandomPositionsKernel << <bloodCellsThreads.blocks, bloodCellsThreads.threadsPerBlock, 0, streams[i] >> >
				(devStates, bloodCells.particles, cylinderBaseCenter, bloodCellModel, bloodCellSize);
			/*gatherForcesKernel<BloodCellDefinition::count, BloodCellDefinition::particlesInCell, particlesStart, graphStart>
				<< <threads.blocks, threads.threadsPerBlock, 0, streams[i] >> > (*this);*/
		});


		HANDLE_ERROR(cudaFree(devStates));
	}

	__global__ void setupCurandStatesKernel(curandState* states, unsigned long seed)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= particleCount)
			return;
		curand_init(seed, id, 0, &states[id]);
	}

	// Generate random positions and velocities at the beginning
	__global__ void generateRandomPositionsKernel(curandState* states, Particles particles, glm::vec3 cylinderBaseCenter, 
		cudaVec3 bloodCellModelPosition, unsigned int bloodCellModelSize/*, float cylinderRadius, float cylinderHeight*/)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= particleCount)
			return;

		if (!(id % bloodCellModelSize)) {
			particles.positions.x[id] = cylinderBaseCenter.x - cylinderRadius * 0.5f + curand_uniform(&states[id]) * cylinderRadius;
			particles.positions.y[id] = cylinderBaseCenter.y - cylinderRadius * 0.5f + curand_uniform(&states[id]) * cylinderRadius + cylinderHeight / 2;
			particles.positions.z[id] = cylinderBaseCenter.z - cylinderRadius * 0.5f + curand_uniform(&states[id]) * cylinderRadius;
		}
		else {
			particles.positions.x[id] = particles.positions.x[id / bloodCellModelSize] + bloodCellModelPosition.x[id % bloodCellModelSize] - bloodCellModelPosition.x[0];
			particles.positions.x[id] = particles.positions.x[id / bloodCellModelSize] + bloodCellModelPosition.y[id % bloodCellModelSize] - bloodCellModelPosition.y[0];
			particles.positions.x[id] = particles.positions.x[id / bloodCellModelSize] + bloodCellModelPosition.z[id % bloodCellModelSize] - bloodCellModelPosition.z[0];
		}

		particles.velocities.x[id] = 0;
		particles.velocities.y[id] = -10;
		particles.velocities.z[id] = 0;

		particles.forces.x[id] = 0;
		particles.forces.y[id] = 0;
		particles.forces.z[id] = 0;
	}

	// Main simulation function, called every frame
	void SimulationController::calculateNextFrame()
	{
		std::visit([&](auto&& g1, auto&& g2)
			{
				// 1. Calculate grids
				// TODO: possible optimization - these grisds can be calculated simultaneously
				g1->calculateGrid(bloodCells.particles, particleCount);
				g2->calculateGrid(triangles.centers.x, triangles.centers.y, triangles.centers.z, triangles.triangleCount);

				// 2. Detect particle collisions
				calculateParticleCollisions << < bloodCellsThreads.blocks, bloodCellsThreads.threadsPerBlock >> > (bloodCells, *g1);
				HANDLE_ERROR(cudaPeekAtLastError());

				// 3. Propagate particle forces into neighbors

				bloodCells.gatherForcesFromNeighbors(streams);
				HANDLE_ERROR(cudaPeekAtLastError());
    
				// 4. Detect vein collisions and propagate forces -> velocities, velocities -> positions for particles

				detectVeinCollisionsAndPropagateParticles << < bloodCellsThreads.blocks, bloodCellsThreads.threadsPerBlock >> > (bloodCells, triangles, *g2);
				HANDLE_ERROR(cudaPeekAtLastError());
    
				// 5. Propagate triangle forces into neighbors

				triangles.gatherForcesFromNeighbors(veinVerticesThreads.blocks, veinVerticesThreads.threadsPerBlock);
				HANDLE_ERROR(cudaPeekAtLastError());
    
				// 6. Propagate forces -> velocities, velocities -> positions for vein triangles
				triangles.propagateForcesIntoPositions(veinVerticesThreads.blocks, veinVerticesThreads.threadsPerBlock);
				HANDLE_ERROR(cudaPeekAtLastError());
    
				// 7. Recalculate triangles centers
				triangles.calculateCenters(veinTrianglesThreads.blocks, veinTrianglesThreads.threadsPerBlock);
				HANDLE_ERROR(cudaPeekAtLastError());

				if constexpr (useBloodFlow)
				{
					HandleVeinEnd(bloodCells, streams);
					HANDLE_ERROR(cudaPeekAtLastError());
				}

			}, particleGrid, triangleGrid);
	}
}