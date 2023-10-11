#ifndef SIMULATION_H
#define SIMULATION_H

#include "../utilities/cuda_threads.hpp"
#include "../objects/blood_cells.cuh"
#include "../objects/vein_triangles.cuh"
#include "../grids/uniform_grid.cuh"
#include "../grids/no_grid.cuh"
#include "vein_end.cuh"

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
		~SimulationController();

		void calculateNextFrame();
	private:
		BloodCells& bloodCells;
		VeinTriangles& triangles;
		Grid particleGrid;
		Grid triangleGrid;

		CudaThreads bloodCellsThreads;
		CudaThreads veinVerticesThreads;
		CudaThreads veinTrianglesThreads;

		cudaStream_t particleGridStream;
		cudaStream_t trianglesGridStream;

		EndVeinHandler endVeinHandler;

		void generateRandomPositions();
	};
}

#endif