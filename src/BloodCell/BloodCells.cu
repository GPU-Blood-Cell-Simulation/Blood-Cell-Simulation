#include "BloodCells.cuh"

#include "cuda_runtime.h"
#include "../utilities.cuh"
#include "../defines.cuh"

BloodCells::BloodCells(int cellsCnt, int particlsInCells, float* graphDesc):
	particles(cellsCnt * particlsInCells)
{
	this->particlesInCell = particlsInCells;
	this->particlesCnt = cellsCnt * particlesInCell;

	int graphSize = particlesInCell * particlesInCell;

	HANDLE_ERROR(cudaMalloc(&springsGrah, sizeof(float) * graphSize));
	HANDLE_ERROR(cudaMemcpy(springsGrah, graphDesc, sizeof(float) * graphSize, cudaMemcpyHostToDevice));
}


void BloodCells::Deallocate()
{
	HANDLE_ERROR(cudaFree(springsGrah));
	particles.freeParticles();
}


__global__ void PropagateForcesOnDevice(BloodCells cells);


void BloodCells::PropagateForces()
{
	int threadsPerBlock = particlesCnt > 1024 ? 1024 : particlesCnt;
	int blocks = (particlesCnt + threadsPerBlock - 1) / threadsPerBlock;

	PropagateForcesOnDevice << <blocks, threadsPerBlock >> > (*this);
}


__device__ inline float CalculateSpringForce(float3 p1, float3 p2, float3 v1, float3 v2, float springLen) {
	return (length(p1 - p2) - springLen) * k_sniff + dot(normalize(p1 - p2), (v1 - v2)) * d_fact;
}


__global__ void PropagateForcesOnDevice(BloodCells cells)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int inCellIndex = index % cells.particlesInCell;

	if (index >= cells.particlesCnt)
		return;

	float3 pos = cells.particles.position.get(index);
	float3 velo = cells.particles.velocity.get(index);

	for (int neighbourCellindex = 0; neighbourCellindex < cells.particlesInCell; neighbourCellindex++)
	{
		float springLen = cells.springsGrah[inCellIndex * cells.particlesInCell + neighbourCellindex];

		if (springLen == NO_SPRING)
			continue;

		int neighbourIndex = index - inCellIndex + neighbourCellindex;

		float3 neighbourPos = cells.particles.position.get(neighbourIndex);
		float3 neighbourVelo = cells.particles.velocity.get(neighbourIndex);

		float springForce = CalculateSpringForce(pos, neighbourPos, velo, neighbourVelo, springLen);

		cells.particles.force.add(index, springForce * normalize(neighbourPos - pos));
	}
}



