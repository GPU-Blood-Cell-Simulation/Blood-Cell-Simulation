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
	class SimulationController
	{
	public:
		SimulationController(BloodCells& bloodCells, DeviceTriangles& triangles, Grid grid);

		void calculateNextFrame();
	private:
		BloodCells& bloodCells;
		DeviceTriangles& triangles;
		Grid grid;

		void generateRandomPositions();
	};
}

#endif