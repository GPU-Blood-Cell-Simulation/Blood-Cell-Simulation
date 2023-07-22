#ifndef SIMULATION_H
#define SIMULATION_H

#include "../blood_cell_structures/blood_cells.cuh"
#include "../blood_cell_structures/device_triangles.cuh"
#include "../grids/uniform_grid.cuh"
#include "../grids/no_grid.cuh"

#include <curand.h>
#include <curand_kernel.h>
#include <variant>

using Grid = std::variant<UniformGrid*, NoGrid*>;

namespace sim
{
	void generateRandomPositions(Particles& particles, const int particleCount);

	void calculateNextFrame(BloodCells& cells, DeviceTriangles& triangles, Grid grid, unsigned int triangleCount);
}

#endif