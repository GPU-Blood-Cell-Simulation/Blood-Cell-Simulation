#include "../blood_cell_structures/blood_cells.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace sim
{
	__global__ void detectParticleCollisions(BloodCells cells, unsigned int* cellIds, unsigned int* particleIds,
		unsigned int* cellStarts, unsigned int* cellEnds);
}