#pragma once

#include <curand.h>
#include <curand_kernel.h>
#include "objects.cuh"

namespace sim
{
	void allocateMemory(unsigned int** cellIds, unsigned int** particleIds, unsigned int** cellStarts, unsigned int** cellEnds,
		const unsigned int particleCount);

	void generateRandomPositions(float* positionX, float* positionY, float* positionZ, const int particleCount);
	void generateInitialPositionsInLayers(particles& p, corpuscles& c, int particleCount, int layersCount);

	void calculateNextFrame(particles& p, corpuscles& c, unsigned int* cellIds, unsigned int* particleIds,
		unsigned int* cellStarts, unsigned int* cellEnds, unsigned int particleCount);
}