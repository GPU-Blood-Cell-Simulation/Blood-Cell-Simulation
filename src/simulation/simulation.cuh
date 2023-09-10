#ifndef SIMULATION_H
#define SIMULATION_H

#include "../objects/blood_cells.cuh"
#include "../objects/vein_triangles.cuh"
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
		SimulationController(BloodCells& bloodCells, VeinTriangles& triangles, Grid particleGrid, Grid triangleGrid);

		void calculateNextFrame();
	private:
		BloodCells& bloodCells;
		VeinTriangles& triangles;
		Grid particleGrid;
		Grid triangleGrid;

		const int bloodCellsThreadsPerBlock;
		const int bloodCellsBlocks;

		const int veinVerticesThreadsPerBlock;
		const int veinVerticesBlocks;

		const int veinTrianglesThreadsPerBlock;
		const int veinTrianglesBlocks;

		void generateRandomPositions();
	};
}

#endif