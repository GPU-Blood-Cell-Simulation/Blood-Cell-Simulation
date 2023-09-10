#include "vein_end.cuh"
#include "../defines.hpp"
#include "../utilities/constexpr_functions.hpp"

#include "cuda_runtime.h"

#include "cstdio"

static enum SynchronizationType { warpSync, blockSync };

constexpr SynchronizationType veinEndSyncType = blockSync;
	//(CudaThreads::threadsInWarp % particlesInBloodCell == 0) ? warpSync : blockSync;

constexpr int CalculateThreadsPerBlock()
{
	switch (veinEndSyncType)
	{
	case warpSync:
		// Max number of full warps
		return (CudaThreads::maxThreadsInBlock / CudaThreads::threadsInWarp) * CudaThreads::threadsInWarp;

	case blockSync:
		// Max mulitple of number of particles in blood cell
		return (CudaThreads::maxThreadsInBlock / particlesInBloodCell) * particlesInBloodCell;

	default:
		throw std::domain_error("Unknown synchronization type");
	}
}

constexpr int CalculateBlocksCount()
{
	return ceilToInt(static_cast<float>(particleCount) / CalculateThreadsPerBlock());
}


EndVeinHandler::EndVeinHandler():
	threads(CalculateThreadsPerBlock(), CalculateBlocksCount())
{}


__global__ void handleVeinEnds(BloodCells bloodCells)
{
	__shared__ bool belowVein[CalculateThreadsPerBlock()];
	int particleId = blockDim.x * blockIdx.x + threadIdx.x;
	 
	if (particleId >= particleCount)
		return;

	float posY = bloodCells.particles.positions.y[particleId]; 

	if (posY >= 0.95f * height) {
		// Bounce particle off upper bound
		bloodCells.particles.velocities.y[particleId] -= 5;
	}

	// Check lower bound
	bool teleport = (posY <= 0.2f*height);
	belowVein[threadIdx.x] = teleport;

	__syncthreads();

	int particleInCellIndex = particleId % particlesInBloodCell;
	int numberOfParticlesInThread = threadIdx.x / particlesInBloodCell * particlesInBloodCell;

	// Algorithm goes through all neighbours and checks if any of them is low enought to be teleported
	#pragma unroll
	for (int i = 1; i < particlesInBloodCell; i++)
	{
 		teleport |= belowVein[((particleInCellIndex + i) % particlesInBloodCell) + numberOfParticlesInThread];
	}

	if (teleport)
	{
		// TODO: add some randomnes to velocity and change positivon to one which is always inside the vein
		bloodCells.particles.positions.y[particleId] = 0.85f * height;
		bloodCells.particles.velocities.set(particleId, make_float3(initVelocityX, initVelocityY, initVelocityZ));
	}
}


void EndVeinHandler::Handle(BloodCells& cells)
{
	handleVeinEnds << <threads.blocks, threads.threadsPerBlock >> > (cells);
}
