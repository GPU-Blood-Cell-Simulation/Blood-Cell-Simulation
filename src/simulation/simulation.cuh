#ifndef SIMULATION_H
#define SIMULATION_H

#include <curand.h>
#include <curand_kernel.h>

#include "../grids/uniform_grid.cuh"
#include "../blood_cell_structures/blood_cells.cuh"
#include "../blood_cell_structures/device_triangles.cuh"


namespace sim
{
	void generateRandomPositions(Particles& particles, const int particleCount);

	void calculateNextFrame(BloodCells& cells, DeviceTriangles& triangles, UniformGrid& grid, unsigned int triangleCount);
}

#endif