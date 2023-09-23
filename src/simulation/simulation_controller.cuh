#ifndef SIMULATION_H
#define SIMULATION_H

#include "../grids/no_grid.cuh"
#include "../grids/uniform_grid.cuh"
#include "../objects/blood_cells.cuh"
#include "../objects/vein_triangles.cuh"
#include "../utilities/cuda_threads.hpp"
#include "vein_end.cuh"

#include <variant>


using Grid = std::variant<UniformGrid*, NoGrid*>;

namespace sim
{
	class SimulationController
	{
	public:
		SimulationController(BloodCells& bloodCells, VeinTriangles& triangles, Grid particleGrid, Grid triangleGrid);

		void calculateNextFrame();
	private:
		BloodCells& bloodCells;
		VeinTriangles& triangles;
		Grid particleGrid;
		Grid triangleGrid;

		CudaThreads bloodCellsThreads;
		CudaThreads veinVerticesThreads;
		CudaThreads veinTrianglesThreads;

		EndVeinHandler endVeinHandler;

		void generateRandomPositions();
	};
}

#endif