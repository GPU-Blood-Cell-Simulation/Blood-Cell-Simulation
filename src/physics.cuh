#include "objects.cuh"
#include "utilities.cuh"
#include <cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>


namespace physics
{
	__global__ void propagateParticles(particles& gp, corpuscles& c, int particleCount)
	{
		// depends on which cell model we use, for dipol 2
		int cell_size = 2;

		int part_index = blockDim.x * blockIdx.x + threadIdx.x;
		int cell_index = part_index / cell_size;

		if (part_index + 1 >= particleCount)
			return;


		// propagate force into velocities
		float3 F = gp.force.get(part_index);
		gp.velocity.add(part_index, dt * F);

		// propagate velocities into positions
		float3 v = gp.velocity.get(part_index);
		float3 dpos = dt * v - gp.position.get(part_index);
		gp.position.add(part_index, dt * v);

		// zero forces
		gp.force.set(part_index, make_float3(0, 0, 0));


		/// must sync here probably
		__syncthreads();

		c.propagateForces(gp, part_index);
	}
}