#pragma once

#include <curand.h>
#include <curand_kernel.h>
#include "objects.cuh"
#include "uniform_grid.cuh"

namespace sim
{
	void generateRandomPositions(Particles particles, const int particleCount);
	void generateInitialPositionsInLayers(Particles particles, Corpuscles corpuscles, int particleCount, int layersCount);

	void calculateNextFrame(Particles p, Corpuscles c, UniformGrid& grid, unsigned int particleCount);
}