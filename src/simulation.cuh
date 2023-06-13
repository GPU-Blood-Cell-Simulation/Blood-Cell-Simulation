#pragma once

#include <curand.h>
#include <curand_kernel.h>
#include "objects.cuh"
#include "uniform_grid.cuh"

namespace sim
{
	void allocateMemory(UniformGrid& grid, const unsigned int particleCount);
	void generateRandomPositions(Particles particles, const int particleCount);
	void generateInitialPositionsInLayers(Particles particles, Corpuscles corpuscles, int particleCount, int layersCount);

	void calculateNextFrame(Particles p, Corpuscles c, Triangles triangles, UniformGrid& grid, unsigned int particleCount);
}