#ifndef SIMULATION_H
#define SIMULATION_H

#include <curand.h>
#include <curand_kernel.h>

#include "objects.cuh"
#include "uniform_grid.cuh"
#include "BloodCell/BloodCells.cuh"


namespace sim
{
	void allocateMemory(UniformGrid& grid, const unsigned int particleCount);
	void generateRandomPositions(Particles particles, const int particleCount);

	void calculateNextFrame(BloodCells cells, DeviceTriangles triangles, UniformGrid& grid, unsigned int trianglesCount);
}

#endif