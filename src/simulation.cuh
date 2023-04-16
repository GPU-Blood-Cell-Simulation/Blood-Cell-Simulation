#pragma once

#include <curand.h>
#include <curand_kernel.h>

namespace sim
{
	void allocateMemory(unsigned int** cellIds, unsigned int** particleIds, unsigned int** cellStarts, unsigned int** cellEnds,
		const unsigned int particleCount);

	void generateRandomPositions(float* positionX, float* positionY, float* positionZ, const int particleCount);
	void generateInitialPositionsInLayers(particles& p, corpuscles& c, const int particleCount, const int layersCount);

	void calculateNextFrame(float* positionX,float* positionY, float* positionZ,
		unsigned int* cellIds, unsigned int* particleIds,
		unsigned int* cellStarts, unsigned int* cellEnds, unsigned int particleCount);
}