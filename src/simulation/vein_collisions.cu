#include "vein_collisions.cuh"

namespace sim
{
	__device__ ray::ray(float3 origin, float3 direction) : origin(origin), direction(direction) {}

	// 1. Calculate collisions between particles and vein triangles
	// 2. Propagate forces into velocities and velocities into positions. Reset forces to 0 afterwards
	__global__ void detectVeinCollisionsAndPropagateParticles(BloodCells cells, DeviceTriangles triangles)
	{
		int part_index = blockDim.x * blockIdx.x + threadIdx.x;

		if (part_index >= cells.particleCount)
			return;

		// propagate force into velocities
		float3 F = cells.particles.force.get(part_index);
		float3 velocity = cells.particles.velocity.get(part_index);
		float3 pos = cells.particles.position.get(part_index);

		// upper and lower bound
		if (pos.y >= 0.9f * height)
			velocity.y -= 5;

		if (pos.y <= 0.1f * height)
			velocity.y += 5;

		velocity = velocity + dt * F;
		ray r(pos, normalize(velocity));
		// collisions with vein cylinder
		// TODO: this is a naive (no grid) implementation
		if (
			calculateSideCollisions(pos, r, triangles) &&
			length(pos - (pos + r.t * r.direction)) <= 5.0f)
		{
			// triangles move vector
			float3 ds = 2 * normalize(make_float3(velocity.x, 0, velocity.z));

			// here velocity should be changed in every direction but triangle normal
			// 0.8 to slow down after contact
			velocity.x *= -0.8;
			velocity.z *= -0.8;

			// move triangle a bit
			triangles.add(r.objectIndex, 0, ds);
			triangles.add(r.objectIndex, 1, ds);
			triangles.add(r.objectIndex, 2, ds);
		}

		cells.particles.velocity.set(part_index, velocity);

		// propagate velocities into positions
		cells.particles.position.add(part_index, dt * velocity);

		// zero forces
		cells.particles.force.set(part_index, make_float3(0, 0, 0));
	}

	// Calculate whether a collision between a particle (represented by the ray) and a vein triangle occurred
	__device__ bool calculateSideCollisions(float3 p, ray& r, DeviceTriangles& triangles)
	{
		for (int i = 0; i < triangles.triangleCount; ++i)
		{
			constexpr float EPS = 0.000001f;
			float3 v1 = triangles.get(i, 0);
			float3 v2 = triangles.get(i, 1);
			float3 v3 = triangles.get(i, 2);

			const float3 edge1 = v2 - v1;
			const float3 edge2 = v3 - v1;
			const float3 h = cross(r.direction, edge2);
			const float a = dot(edge1, h);
			if (a > -EPS && a < EPS)
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
			if (t > EPS)
			{
				r.t = t;
				r.objectIndex = i;
				return true;
			}
		}
		return false;
	}
}