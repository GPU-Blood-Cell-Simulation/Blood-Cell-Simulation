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

		// collisions with vein cylinder
		if ((pos.x - width/2) * (pos.x - width/2)+ (pos.z - depth/2) * (pos.z - depth/2) >= 
			cylinderScaleX * cylinderScaleX * cylinderRadius * cylinderRadius)
		{
			velocity.x *= -1;
			velocity.z *= -1;
		}

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