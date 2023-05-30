#pragma once

#include <curand.h>
#include <curand_kernel.h>
#include "objects.cuh"

namespace sim
{
	void allocateMemory(unsigned int** cellIds, unsigned int** particleIds, unsigned int** cellStarts, unsigned int** cellEnds,
		const unsigned int particleCount);

	void generateRandomPositions(Particles particles, const int particleCount);
	void generateInitialPositionsInLayers(Particles particles, Corpuscles corpuscles, int particleCount, int layersCount);

	void calculateNextFrame(Particles p, Corpuscles c, unsigned int* cellIds, unsigned int* particleIds,
		unsigned int* cellStarts, unsigned int* cellEnds, unsigned int particleCount);

	void deallocateMemory(unsigned int* cellIds, unsigned int* particleIds, unsigned int* cellStarts, unsigned int* cellEnds);
}