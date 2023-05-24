#include "objects.cuh"
#include "utilities.cuh"
#include "BloodCell/BloodCells.cuh"

#include <cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>


namespace physics
{
	__global__ void propagateParticles(BloodCells cells)
	{
		int part_index = blockDim.x * blockIdx.x + threadIdx.x;
		int cell_index = part_index / cells.particlesInCell;

		if (part_index >= cells.particlesCnt)
			return;

		// propagate force into velocities
		float3 F = cells.particles.force.get(part_index);
		float3 velocity = cells.particles.velocity.get(part_index);
		float3 pos = cells.particles.position.get(part_index);

		velocity = velocity + dt * F;

		// collisions with vein cylinder
		if ((pos.x - width/2) * (pos.x - width/2)+ (pos.z - depth/2) * (pos.z - depth/2) >= 
			cylinderScaleX * cylinderScaleX * cylinderRadius * cylinderRadius)
		{
			velocity.x *= -1;
			velocity.z *= -1;
		}

		cells.particles.velocity.set(part_index, velocity);

		// propagate velocities into positions
		float3 dpos = dt * velocity - cells.particles.position.get(part_index);
		cells.particles.position.add(part_index, dt * velocity);

		// zero forces
		cells.particles.force.set(part_index, make_float3(0, 0, 0));
	}
}