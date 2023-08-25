#ifndef SIMULATION_H
#define SIMULATION_H

#include "../objects/blood_cells.cuh"
#include "../objects/device_triangles.cuh"
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
		SimulationController(BloodCells& bloodCells, DeviceTriangles& triangles, Grid particleGrid, Grid triangleGrid);

		void calculateNextFrame();
	private:
		BloodCells& bloodCells;
		DeviceTriangles& triangles;
		Grid particleGrid;
		Grid triangleGrid;

		void generateRandomPositions();
	};
}

#endif