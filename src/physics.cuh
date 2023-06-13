#pragma once
#include "objects.cuh"
#include "utilities.cuh"
#include <cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>


namespace physics
{
	__global__ void propagateParticles(Particles particles, Corpuscles corpuscles, Triangles triangles, int particleCount, int trianglesCount);

	__device__ bool calculateSideCollisions(float3 p, ray& r, Triangles triangles, int trianglesCount);

	__global__ void propagateParticles(Particles particles, Corpuscles corpuscles, Triangles triangles, int particleCount, int trianglesCount)
	{
		// depends on which cell model we use, for dipole 2
		int cell_size = 2;

		int part_index = blockDim.x * blockIdx.x + threadIdx.x;
		int cell_index = part_index / cell_size;

		if (part_index >= particleCount)
			return;

		// propagate force into velocities
		float3 F = particles.force.get(part_index);
		float3 velocity = particles.velocity.get(part_index);
		float3 pos = particles.position.get(part_index);

		velocity = velocity + dt * F;

		// collisions with vein cylinder
		if (!(velocity.x == 0 && velocity.y == 0 && velocity.z == 0) &&
			calculateSideCollisions(pos, ray(pos, normalize(velocity)), triangles, trianglesCount))
		{
			// here velocity should be changed in every direction but triangle normal
			velocity.x *= -1;
			velocity.z *= -1;
		}

		particles.velocity.set(part_index, velocity);

		// propagate velocities into positions
		float3 dpos = dt * velocity - particles.position.get(part_index);
		particles.position.add(part_index, dt * velocity);

		// zero forces
		particles.force.set(part_index, make_float3(0, 0, 0));


		/// must sync here probably
		__syncthreads();

		corpuscles.propagateForces(particles, part_index);
	}


	__device__ bool calculateSideCollisions(float3 p, ray& r, Triangles triangles, int trianglesCount)
	{
		for (int i = 0; i < trianglesCount; ++i)
		{
			float3 v1 = triangles.v1.get(i);
			float3 v2 = triangles.v2.get(i);
			float3 v3 = triangles.v3.get(i);

			const float3 edge1 = v2 - v1;
			const float3 edge2 = v3 - v1;
			const float3 h = cross(r.direction, edge2);
			const float a = dot(edge1, h);
			if (a > -0.0001f && a < 0.0001f)
				continue; // ray parallel to triangle
			const float f = 1 / a;
			const float3 s = r.origin - v1;
			const float u = f * dot(s, h);
			if (u < 0 || u > 1)
				continue;
			const float3 q = cross(s, edge1);
			const float v = f * dot(r.direction, q);
			if (v < 0 || u + v > 1)
				continue;
			const float t = f * dot(edge2, q);
			if (t > 0.0001f)
			{
				r.t = r.t > t ? t : r.t;
				return true;
			}
		}
		return false;
	}
}