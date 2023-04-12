#include "objects.cuh"

namespace physics
{
	__global__ void propagateParticles(particles& gp, corpuscle* c)
	{
		// depends on which cell model we use
		int cell_size = 2;

		/// TODO
		int Idx = blockDim.x * blockIdx.x + threadIdx.x;

		// not less than one warp per cell
		int warps_per_cell = (int)std::ceil(float(cell_size) / warpSize);

		/// TODO
		int part_index;
		int cell_index = part_index / cell_size;


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

		// propagate positions into spring forces

		// foreach spring TODO

		float* spring = c[cell_index].spr_map.getSpringPtr(part_index);
		int size = (int)*spring;
		for (int i = 0; i < size; i += 2)
		{
			int oth_part = spring[i];
			float L0 = spring[i + 1];
			float3 p1 = gp.position.get(part_index);
			float3 p2 = gp.position.get(oth_part);
			float3 v1 = gp.velocity.get(part_index);
			float3 v2 = gp.velocity.get(oth_part);
			float Fr = (length(p1 - p2) - L0) * k_sniff + dot(normalize(p1 - p2), (v1 - v2)) * d_fact;

			// accumulate forces
			gp.force.add(part_index, Fr * normalize(p2 - p1));
			gp.force.add(oth_part, Fr * normalize(p1 - p2));
		}
	}
}
