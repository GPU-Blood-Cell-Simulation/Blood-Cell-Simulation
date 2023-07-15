#ifndef SIMULATION_H
#define SIMULATION_H

#include "../blood_cell_structures/blood_cells.cuh"
#include "../blood_cell_structures/device_triangles.cuh"
#include "../grids/uniform_grid.cuh"

#include <curand.h>
#include <curand_kernel.h>


namespace sim
{
	void generateRandomPositions(Particles& particles, const int particleCount);

	void calculateNextFrame(BloodCells& cells, DeviceTriangles& triangles, UniformGrid& grid, unsigned int triangleCount);
}

#endif