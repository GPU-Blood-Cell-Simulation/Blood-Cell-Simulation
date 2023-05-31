#include "uniform_grid.cuh"
#include "defines.cuh"

#include <cstdio>
#include <cstdlib>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#pragma region Helper kernels

__global__ void calculateCellIdKernel(const float* positionX, const float* positionY, const float* positionZ,
	unsigned int* CellIds, unsigned int* ParticleIds, const unsigned int particleCount);

__global__ void calculateStartAndEndOfCellKernel(const float* positionX, const float* positionY, const float* positionZ,
	const unsigned int* cellIds, const unsigned int* particleIds,
	unsigned int* cellStarts, unsigned int* cellEnds, unsigned int particleCount);

__global__ void calculateCellIdKernel(const float* positionX, const float* positionY, const float* positionZ,
	unsigned int* cellIds, unsigned int* particleIds, const unsigned int particleCount)
{
	unsigned int particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId >= particleCount)
		return;

	unsigned int cellId =
		static_cast<unsigned int>(positionX[particleId] / cellWidth) +
		static_cast<unsigned int>(positionY[particleId] / cellHeight) +
		static_cast<unsigned int>(positionZ[particleId] / cellDepth);

	particleIds[particleId] = particleId;
	cellIds[particleId] = cellId;

}

__global__ void calculateStartAndEndOfCellKernel(const float* positionX, const float* positionY, const float* positionZ,
	const unsigned int* cellIds, const unsigned int* particleIds,
	unsigned int* cellStarts, unsigned int* cellEnds, unsigned int particleCount)
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= particleCount)
		return;

	unsigned int currentCellId = cellIds[id];

	// Check if the previous cell id was different - it would mean we found the start of a cell
	if (id > 0 && currentCellId != cellIds[id - 1])
	{
		cellStarts[currentCellId] = id;
	}

	// Check if the next cell id was different - it would mean we found the end of a cell
	if (id < particleCount - 1 && currentCellId != cellIds[id + 1])
	{
		cellEnds[currentCellId] = id;
	}

	// debug
	//if (id < 5) printf("%d. particle id: %d, cell: %d\n", id, particleIds[id], cellIds[id]);

}

#pragma endregion

// Allocate GPU buffers for the index buffers
UniformGrid::UniformGrid()
{
	HANDLE_ERROR(cudaMalloc((void**)&cellIds, PARTICLE_COUNT * sizeof(unsigned int)));
	HANDLE_ERROR(cudaMalloc((void**)&particleIds, PARTICLE_COUNT * sizeof(unsigned int)));

	HANDLE_ERROR(cudaMalloc((void**)&cellStarts, width / cellWidth * height / cellHeight * depth / cellDepth * sizeof(unsigned int)));
	HANDLE_ERROR(cudaMalloc((void**)&cellEnds, width / cellWidth * height / cellHeight * depth / cellDepth * sizeof(unsigned int)));
}

UniformGrid::~UniformGrid()
{
	cudaFree(cellIds);
	cudaFree(particleIds);
	cudaFree(cellStarts);
	cudaFree(cellEnds);
}

void UniformGrid::calculateGrid(const Particles& particles)
{
	// Calculate launch parameters

	constexpr int threadsPerBlock = PARTICLE_COUNT > 1024 ? 1024 : PARTICLE_COUNT;
	constexpr int blocks = (PARTICLE_COUNT + threadsPerBlock - 1) / threadsPerBlock;

	// 1. Calculate cell id for every particle and store as pair (cell id, particle id) in two buffers
	calculateCellIdKernel << <blocks, threadsPerBlock >> >
		(particles.position.x, particles.position.y, particles.position.z, cellIds, particleIds, PARTICLE_COUNT);

	// 2. Sort particle ids by cell id

	thrust::device_ptr<unsigned int> keys = thrust::device_pointer_cast<unsigned int>(cellIds);
	thrust::device_ptr<unsigned int> values = thrust::device_pointer_cast<unsigned int>(particleIds);

	thrust::stable_sort_by_key(keys, keys + PARTICLE_COUNT, values);

	// 3. Find the start and end of every cell

	calculateStartAndEndOfCellKernel << <blocks, threadsPerBlock >> >
		(particles.position.x, particles.position.y, particles.position.z,
			cellIds, particleIds, cellStarts, cellEnds,PARTICLE_COUNT);
}