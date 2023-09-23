#include "vein_end.cuh"

#include "../defines.hpp"
#include "../utilities/math.cuh"

#include "cuda_runtime.h"


constexpr float upperBoundTreshold = 0.95f * height;
constexpr float lowerBoundTreshold = 0.2f * height;
constexpr float targetTeleportHeight = 0.85f * height;

enum SynchronizationType { warpSync, blockSync };

constexpr SynchronizationType veinEndSyncType =
	(CudaThreads::threadsInWarp % particlesInBloodCell == 0) ? warpSync : blockSync;

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
	return constCeil(static_cast<float>(particleCount) / CalculateThreadsPerBlock());
}


EndVeinHandler::EndVeinHandler():
	threads(CalculateThreadsPerBlock(), CalculateBlocksCount())
{}


__global__ void handleVeinEndsBlockSync(BloodCells bloodCells)
{
	__shared__ bool belowVein[CalculateThreadsPerBlock()];
	int particleId = blockDim.x * blockIdx.x + threadIdx.x;
	 
	if (particleId >= particleCount)
		return;

	float posY = bloodCells.particles.positions.y[particleId]; 

	if (posY >= upperBoundTreshold) {
		// Bounce particle off upper bound
		bloodCells.particles.velocities.y[particleId] -= 5;
	}

	// Check lower bound
	bool teleport = (posY <= lowerBoundTreshold);
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
		bloodCells.particles.positions.y[particleId] = targetTeleportHeight;
		bloodCells.particles.velocities.set(particleId, make_float3(initVelocityX, initVelocityY, initVelocityZ));
	}
}


constexpr int initSyncBitMask = (particlesInBloodCell == 32) ? 0xffffffff : (1 << (particlesInBloodCell)) - 1;


__global__ void handleVeinEndsWarpSync(BloodCells bloodCells)
{
	int particleId = blockDim.x * blockIdx.x + threadIdx.x;
	if (particleId >= particleCount)
		return;

	int threadInWarpID = threadIdx.x % CudaThreads::threadsInWarp;
	float posY = bloodCells.particles.positions.y[particleId];

	if (posY >= upperBoundTreshold) {
		// Bounce particle off upper bound
		bloodCells.particles.velocities.y[particleId] -= 5;
	}

	int syncBitMask = initSyncBitMask << static_cast<int>(std::floor(static_cast<float>(threadInWarpID)/particlesInBloodCell)) * particlesInBloodCell;

	// Bit mask of particles, which are below treshold
	int particlesBelowTreshold = __any_sync(syncBitMask, posY <= lowerBoundTreshold);

	if (particlesBelowTreshold != 0) {
		// TODO: add some randomnes to velocity and change positivon to one which is always inside the vein
		bloodCells.particles.positions.y[particleId] = targetTeleportHeight;
		bloodCells.particles.velocities.set(particleId, make_float3(initVelocityX, initVelocityY, initVelocityZ));
	}
}


void EndVeinHandler::Handle(BloodCells& cells)
{ 
	if constexpr (veinEndSyncType == warpSync)
		handleVeinEndsWarpSync << <threads.blocks, threads.threadsPerBlock >> > (cells);
	else if constexpr (veinEndSyncType == blockSync)
		handleVeinEndsBlockSync << <threads.blocks, threads.threadsPerBlock >> > (cells);
	else
		throw std::domain_error("Unknown synchronization type");
}
