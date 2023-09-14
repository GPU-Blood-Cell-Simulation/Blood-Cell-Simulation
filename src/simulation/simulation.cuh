#ifndef SIMULATION_H
#define SIMULATION_H

#include "../objects/blood_cells.cuh"
#include "../objects/vein_triangles.cuh"
#include "../grids/uniform_grid.cuh"
#include "../grids/no_grid.cuh"

#include <curand.h>
#include <curand_kernel.h>
#include <variant>

using Grid = std::variant<UniformGrid*, NoGrid*, OctreeGrid*>

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

		const unsigned int bloodCellsThreadsPerBlock;
		const unsigned int bloodCellsBlocks;

		const unsigned int veinVerticesThreadsPerBlock;
		const unsigned int veinVerticesBlocks;

		const unsigned int veinTrianglesThreadsPerBlock;
		const unsigned int veinTrianglesBlocks;

		void generateRandomPositions();
	};
}

#endif