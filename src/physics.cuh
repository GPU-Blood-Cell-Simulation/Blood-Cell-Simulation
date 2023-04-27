#include "objects.cuh"
#include "utilities.cuh"
#include <cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>


namespace physics
{
	__global__ void propagateParticles(particles gp, dipols c, int particleCount)
	{
		// depends on which cell model we use, for dipole 2
		int cell_size = 2;

		int part_index = blockDim.x * blockIdx.x + threadIdx.x;
		int cell_index = part_index / cell_size;

		if (part_index >= particleCount)
			return;

		// propagate force into velocities
		float3 F = gp.force.get(part_index);
		float3 velocity = gp.velocity.get(part_index);
		float3 pos = gp.position.get(part_index);

		velocity = velocity + dt * F;
		
		// flip velocities in case of colliding with a wall
		if (pos.x <= 0 && velocity.x < 0)
			velocity.x *= -1;
		else if (pos.x >= width && velocity.x > 0)
			velocity.x *= -1;
		if (pos.y <= 0 && velocity.y < 0)
			velocity.y *= -1;
		else if (pos.y >= height && velocity.y > 0)
			velocity.y *= -1;;
		if (pos.z <= 0 && velocity.z < 0)
			velocity.z *= -1;
		else if (pos.z >= depth && velocity.z > 0)
			velocity.z *= -1;

		gp.velocity.set(part_index, velocity);

		// propagate velocities into positions
		float3 dpos = dt * velocity - gp.position.get(part_index);
		gp.position.add(part_index, dt * velocity);

		// zero forces
		gp.force.set(part_index, make_float3(0, 0, 0));


		/// must sync here probably
		__syncthreads();

		c.propagateForces(gp, part_index);
	}
}