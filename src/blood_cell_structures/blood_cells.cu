#include "blood_cells.cuh"
#include "../utilities/cuda_handle_error.cuh"
#include "../utilities/math.cuh"
#include "../defines.hpp"

#include <vector>

#include "cuda_runtime.h"


BloodCells::BloodCells(int cellCount, int particlesInCell, const float* graphDesc) :
	particles(cellCount * particlesInCell), particlesInCell(particlesInCell), particleCount(cellCount * particlesInCell)
{
	int graphSize = particlesInCell * particlesInCell;

	HANDLE_ERROR(cudaMalloc(&springsGraph, sizeof(float) * graphSize));
	HANDLE_ERROR(cudaMemcpy(springsGraph, graphDesc, sizeof(float) * graphSize, cudaMemcpyHostToDevice));
}

BloodCells::BloodCells(const BloodCells& other) : isCopy(true), particles(other.particles), particlesInCell(other.particlesInCell),
particleCount(other.particleCount), springsGraph(other.springsGraph) {}


BloodCells::~BloodCells()
{
	if (!isCopy)
	{
		HANDLE_ERROR(cudaFree(springsGraph));
	}
}


__global__ void PropagateForcesOnDevice(BloodCells cells);


void BloodCells::propagateForces()
{
	int threadsPerBlock = particleCount > 1024 ? 1024 : particleCount;
	int blocks = (particleCount + threadsPerBlock - 1) / threadsPerBlock;

	PropagateForcesOnDevice << <blocks, threadsPerBlock >> > (*this);
}


// TODO : split into two kernels with a temporary buffer - necessary synchronization
__global__ void PropagateForcesOnDevice(BloodCells cells)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int inCellIndex = index % cells.particlesInCell;

	if (index >= cells.particleCount)
		return;

	float3 pos = cells.particles.position.get(index);
	float3 velo = cells.particles.velocity.get(index);

	for (int neighbourCellindex = 0; neighbourCellindex < cells.particlesInCell; neighbourCellindex++)
	{
		float springLen = cells.springsGraph[inCellIndex * cells.particlesInCell + neighbourCellindex];

		if (springLen == NO_SPRING)
			continue;

		int neighbourIndex = index - inCellIndex + neighbourCellindex;

		float3 neighbourPos = cells.particles.position.get(neighbourIndex);
		float3 neighbourVelo = cells.particles.velocity.get(neighbourIndex);

		float springForce = cells.calculateParticleSpringForce(pos, neighbourPos, velo, neighbourVelo, springLen);

		cells.particles.force.add(index, springForce * normalize(neighbourPos - pos));
	}
}