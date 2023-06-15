#pragma once
#include "objects.cuh"
#include "utilities.cuh"
#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>


namespace physics
{
	__global__ void propagateParticles(Particles particles, Corpuscles corpuscles, DeviceTriangles triangles, int particleCount, int trianglesCount);

	__device__ bool calculateSideCollisions(float3 p, ray& r, DeviceTriangles triangles, int trianglesCount);

	__global__ void propagateParticles(Particles particles, Corpuscles corpuscles, DeviceTriangles triangles, int particleCount, int trianglesCount)
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
		ray r(pos, normalize(velocity));
		// collisions with vein cylinder
		if (
			//!(velocity.x == 0 && velocity.y == 0 && velocity.z == 0) &&
			calculateSideCollisions(pos, r, triangles, trianglesCount))
		{
			// triangles move vector
			float3 ds = 1 * normalize(make_float3(velocity.x, 0, velocity.z));

			// here velocity should be changed in every direction but triangle normal
			// 0.8 to slow down after contact
			velocity.x *= -0.8;
			velocity.z *= -0.8;

			// move triangle a bit
			triangles.add(r.objectIndex, 0, ds);
			triangles.add(r.objectIndex, 1, ds);
			triangles.add(r.objectIndex, 2, ds);
		}

		particles.velocity.set(part_index, velocity);

		// upper and lower bound
		/*if (pos.y >= height)
			velocity.y -= 30;

		if (pos.y <= 0)
			velocity.y += 30;*/

		// propagate velocities into positions
		//float3 dpos = dt * velocity - particles.position.get(part_index);
		particles.position.add(part_index, dt * velocity);

		// zero forces
		particles.force.set(part_index, make_float3(0, 0, 0));


		/// must sync here probably
		__syncthreads();

		corpuscles.propagateForces(particles, part_index);
	}

	__device__ bool calculateSideCollisions(float3 p, ray& r, DeviceTriangles triangles, int trianglesCount)
	{
		for (int i = 0; i < trianglesCount; ++i)
		{
			const float EPS = 0.000001f;
			float3 v1 = triangles.get(i,0);
			float3 v2 = triangles.get(i,1);
			float3 v3 = triangles.get(i,2);

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
				r.t = r.t > t ? t : r.t;
				r.objectIndex = i;
				return true;
			}
		}
		return false;

		//for (int i = 0; i < trianglesCount; ++i)
		//{
		//	const float EPS = 0.000001f;
		//	float3 v1 = triangles.get(i, 0);
		//	float3 v2 = triangles.get(i, 1);
		//	float3 v3 = triangles.get(i, 2);
		//	const float3 edge2 = v2 - v1;
		//	const float3 edge1 = v3 - v1;
		//	const float3 normal = -1 * normalize(cross(edge1, edge2));

		//	/*const float dist = dot(normal, p - v1);
		//	if (dist > 10.0f)
		//		continue;*/

		//	const float a = -1 * dot(r.direction, normal);
		//	if (a > -EPS && a < EPS)
		//		continue; // ray parallel to triangle 
		//	const float3 s = r.origin - v1;
		//	const float f = 1 / a;
		//	const float t = f * dot(s, normal);
		//	if (t < EPS)
		//		continue;
		//	const float3 q = cross(s, r.direction);
		//	const float u = f * dot(edge2, q);
		//	const float v = -1 * f * dot(edge1, q);
		//	if (v < 0 || u + v > 1)
		//		continue;

		//	r.t = r.t > t ? t : r.t;
		//	r.objectIndex = i;
		//	return true;
		//}
		//return false;
	}
}
	