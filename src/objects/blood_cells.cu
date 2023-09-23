#include "blood_cells.cuh"

#include "../defines.hpp"
#include "../utilities/cuda_handle_error.cuh"
#include "../utilities/math.cuh"

#include <vector>

#include "cuda_runtime.h"


BloodCells::BloodCells(int cellCount, int particlesInCell, const float* graphDesc) :
	particles(cellCount * particlesInCell)
{
	int graphSize = particlesInCell * particlesInCell;

	HANDLE_ERROR(cudaMalloc(&springsGraph, sizeof(float) * graphSize));
	HANDLE_ERROR(cudaMemcpy(springsGraph, graphDesc, sizeof(float) * graphSize, cudaMemcpyHostToDevice));
}

BloodCells::BloodCells(const BloodCells& other) : isCopy(true), particles(other.particles), springsGraph(other.springsGraph) {}


BloodCells::~BloodCells()
{
	if (!isCopy)
	{
		HANDLE_ERROR(cudaFree(springsGraph));
	}
}


__global__ static void gatherForcesKernel(BloodCells cells);


void BloodCells::gatherForcesFromNeighbors(int blocks, int threadsPerBlock)
{
	gatherForcesKernel << <blocks, threadsPerBlock >> > (*this);
}


__global__ static void gatherForcesKernel(BloodCells cells)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int inCellIndex = index % particlesInBloodCell;

	if (index >= particleCount)
		return;

	float3 pos = cells.particles.positions.get(index);
	float3 velo = cells.particles.velocities.get(index);

	for (int neighbourCellindex = 0; neighbourCellindex < particlesInBloodCell; neighbourCellindex++)
	{
		float springLen = cells.springsGraph[inCellIndex * particlesInBloodCell + neighbourCellindex];

		if (springLen == NO_SPRING)
			continue;

		int neighbourIndex = index - inCellIndex + neighbourCellindex;

		float3 neighbourPos = cells.particles.positions.get(neighbourIndex);
		float3 neighbourVelo = cells.particles.velocities.get(neighbourIndex);

		float springForce = cells.calculateParticleSpringForce(pos, neighbourPos, velo, neighbourVelo, springLen);

		cells.particles.forces.add(index, springForce * normalize(neighbourPos - pos));
	}
}
