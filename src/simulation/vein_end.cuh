#pragma once

#include "../objects/cuda_threads.hpp"
#include "../objects/blood_cells.cuh"


class EndVeinHandler
{
public:
	EndVeinHandler();

	void Handle(BloodCells& cells);

private:
	CudaThreads threads;
};
