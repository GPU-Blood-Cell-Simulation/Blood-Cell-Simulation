#include "vein_collisions.cuh"
#include "../utilities/vertex_index_enum.h"


namespace sim
{
	__device__ ray::ray(float3 origin, float3 direction) : origin(origin), direction(direction) {}

	__device__ bool realCollisionDetection(float3 v0, float3 v1, float3 v2, ray& velocityRay, float3& reflectionVector)
	{
		constexpr float EPS = 0.000001f;
		const float3 edge1 = v1 - v0;
		const float3 edge2 = v2 - v0;

		const float3 h = cross(velocityRay.direction, edge2);
		const float a = dot(edge1, h);
		if (a > -EPS && a < EPS)
			return false; // ray parallel to triangle

		const float f = 1 / a;
		const float3 s = velocityRay.origin - v0;
		const float u = f * dot(s, h);
		if (u < 0 || u > 1)
			return false;
		const float3 q = cross(s, edge1);
		const float v = f * dot(velocityRay.direction, q);
		if (v < 0 || u + v > 1)
			return false;
		const float t = f * dot(edge2, q);
		if (t > EPS)
		{
			velocityRay.t = t;

			// this normal is oriented to the vein interior
			// it is caused by the order of vertices in triangles used to correct face culling
			// change order of edge2 and edge1 in cross product for oposite normal
			// Question: Is the situation when we should use oposite normal possible ?
			float3 normal = normalize(cross(edge2, edge1));
			reflectionVector = velocityRay.direction - 2 * dot(velocityRay.direction, normal) * normal;
			return true;
		}
	}

	__device__ float3 calculateBaricentric(float3 point, float3 v0, float3 v1, float3 v2)
	{
		float3 baricentric;
		float3 e0 = v1 - v0, e1 = v2 - v1, e2 = point - v0;
		float d00 = dot(e0, e0);
		float d01 = dot(e0, e1);
		float d11 = dot(e1, e1);
		float d20 = dot(e2, e0);
		float d21 = dot(e2, e1);
		float denom = d00 * d11 - d01 * d01;
		baricentric.x = (d11 * d20 - d01 * d21) / denom;
		baricentric.y = (d00 * d21 - d01 * d20) / denom;
		baricentric.z = 1.0f - baricentric.x - baricentric.y;
		return baricentric;
	}



	// 1. Calculate collisions between particles and vein triangles
	// 2. Propagate forces into velocities and velocities into positions. Reset forces to 0 afterwards
	template<>
	__global__ void detectVeinCollisionsAndPropagateParticles<UniformGrid>(BloodCells cells, DeviceTriangles triangles, UniformGrid triangleGrid )
	{
		int particleId = blockDim.x * blockIdx.x + threadIdx.x;

		if (particleId >= cells.particleCount)
			return;

		// propagate force into velocities
		float3 F = cells.particles.force.get(particleId);
		float3 velocity = cells.particles.velocity.get(particleId);
		float3 pos = cells.particles.position.get(particleId);

		// upper and lower bound
		if (pos.y >= 0.9f * height)
			velocity.y -= 5;

		if (pos.y <= 0.1f * height)
			velocity.y += 5;

		velocity = velocity + dt * F;
		float3 velocityDir = normalize(velocity);
		ray r(pos, velocityDir);
		float3 reflectedVelociy = make_float3(0, 0, 0);

		bool collicionOccured = false;
		int xId = static_cast<unsigned int>(cells.particles.position.x[particleId] / cellWidth);
		int yId = static_cast<unsigned int>(cells.particles.position.y[particleId] / cellHeight) * cellCountX;
		int zId = static_cast<unsigned int>(cells.particles.position.z[particleId] / cellDepth) * cellCountX * cellCountY;

		// collisions with vein cylinder
		if (xId < 1)
		{
			if (yId < 1)
			{
				if (zId < 1)
				{
					collicionOccured = calculateSideCollisions<0, 1, 0, 1, 0, 1>(pos, r, reflectedVelociy, triangles, triangleGrid );
				}
				else if (zId > cellCountZ - 2)
				{
					collicionOccured = calculateSideCollisions<0, 1, 0, 1, -1, 0>(pos, r, reflectedVelociy, triangles, triangleGrid );
				}
				else
				{
					collicionOccured = calculateSideCollisions<0, 1, 0, 1, -1, 1>(pos, r, reflectedVelociy, triangles, triangleGrid );
				}
			}
			else if (yId > cellCountY - 2)
			{
				if (zId < 1)
				{
					collicionOccured = calculateSideCollisions<0, 1, -1, 0, 0, 1>(pos, r, reflectedVelociy, triangles, triangleGrid );
				}
				else if (zId > cellCountZ - 2)
				{
					collicionOccured = calculateSideCollisions<0, 1, -1, 0, -1, 0>(pos, r, reflectedVelociy, triangles, triangleGrid );
				}
				else
				{
					collicionOccured = calculateSideCollisions<0, 1, -1, 0, -1, 1>(pos, r, reflectedVelociy, triangles, triangleGrid );
				}
			}
			else
			{
				if (zId < 1)
				{
					collicionOccured = calculateSideCollisions<0, 1, -1, 1, 0, 1>(pos, r, reflectedVelociy, triangles, triangleGrid );
				}
				else if (zId > cellCountZ - 2)
				{
					collicionOccured = calculateSideCollisions<0, 1, -1, 1, -1, 0>(pos, r, reflectedVelociy, triangles, triangleGrid );
				}
				else
				{
					collicionOccured = calculateSideCollisions<0, 1, -1, 1, -1, 1>(pos, r, reflectedVelociy, triangles, triangleGrid );
				}
			}
		}
		else if (xId > cellCountX - 2)
		{
			if (yId < 1)
			{
				if (zId < 1)
				{
					collicionOccured = calculateSideCollisions<-1, 0, 0, 1, 0, 1>(pos, r, reflectedVelociy, triangles, triangleGrid );
				}
				else if (zId > cellCountZ - 2)
				{
					collicionOccured = calculateSideCollisions<-1, 0, 0, 1, -1, 0>(pos, r, reflectedVelociy, triangles, triangleGrid );
				}
				else
				{
					collicionOccured = calculateSideCollisions<-1, 0, 0, 1, -1, 1>(pos, r, reflectedVelociy, triangles, triangleGrid );
				}
			}
			else if (yId > cellCountY - 2)
			{
				if (zId < 1)
				{
					collicionOccured = calculateSideCollisions<-1, 0, -1, 0, 0, 1>(pos, r, reflectedVelociy, triangles, triangleGrid );
				}
				else if (zId > cellCountZ - 2)
				{
					collicionOccured = calculateSideCollisions<-1, 0, -1, 0, -1, 0>(pos, r, reflectedVelociy, triangles, triangleGrid );
				}
				else
				{
					collicionOccured = calculateSideCollisions<-1, 0, -1, 0, -1, 1>(pos, r, reflectedVelociy, triangles, triangleGrid );
				}
			}
			else
			{
				if (zId < 1)
				{
					collicionOccured = calculateSideCollisions<-1, 0, -1, 1, 0, 1>(pos, r, reflectedVelociy, triangles, triangleGrid );
				}
				else if (zId > cellCountZ - 2)
				{
					collicionOccured = calculateSideCollisions<-1, 0, -1, 1, -1, 0>(pos, r, reflectedVelociy, triangles, triangleGrid );
				}
				else
				{
					collicionOccured = calculateSideCollisions<-1, 0, -1, 1, -1, 1>(pos, r, reflectedVelociy, triangles, triangleGrid );
				}
			}
		}
		else
		{
			if (yId < 1)
			{
				if (zId < 1)
				{
					collicionOccured = calculateSideCollisions<-1, 1, 0, 1, 0, 1>(pos, r, reflectedVelociy, triangles, triangleGrid );
				}
				else if (zId > cellCountZ - 2)
				{
					collicionOccured = calculateSideCollisions<-1, 1, 0, 1, -1, 0>(pos, r, reflectedVelociy, triangles, triangleGrid );
				}
				else
				{
					collicionOccured = calculateSideCollisions<-1, 1, 0, 1, -1, 1>(pos, r, reflectedVelociy, triangles, triangleGrid );
				}
			}
			else if (yId > cellCountY - 2)
			{
				if (zId < 1)
				{
					collicionOccured = calculateSideCollisions<-1, 1, -1, 0, 0, 1>(pos, r, reflectedVelociy, triangles, triangleGrid );
				}
				else if (zId > cellCountZ - 2)
				{
					collicionOccured = calculateSideCollisions<-1, 1, -1, 0, -1, 0>(pos, r, reflectedVelociy, triangles, triangleGrid );
				}
				else
				{
					collicionOccured = calculateSideCollisions<-1, 1, -1, 0, -1, 1>(pos, r, reflectedVelociy, triangles, triangleGrid );
				}
			}
			else
			{
				if (zId < 1)
				{
					collicionOccured = calculateSideCollisions<-1, 1, -1, 1, 0, 1>(pos, r, reflectedVelociy, triangles, triangleGrid );
				}
				else if (zId > cellCountZ - 2)
				{
					collicionOccured = calculateSideCollisions<-1, 1, -1, 1, -1, 0>(pos, r, reflectedVelociy, triangles, triangleGrid );
				}
				else
				{
					collicionOccured = calculateSideCollisions<-1, 1, -1, 1, -1, 1>(pos, r, reflectedVelociy, triangles, triangleGrid );
				}
			}
		}

		if (collicionOccured)
		{
			// triangles move vector, 2 is experimentall constant
			float3 ds = 0.8f * velocityDir;

			float speed = length(velocity);
			velocity = velocityCollisionDamping * speed * reflectedVelociy;

			float3 v0 = triangles.get(r.objectIndex, vertex0);
			float3 v1 = triangles.get(r.objectIndex, vertex1);
			float3 v2 = triangles.get(r.objectIndex, vertex2);
			float3 baricentric = calculateBaricentric(pos + r.t * r.direction, v0, v1, v2);

			// move triangle a bit
			triangles.add(r.objectIndex, vertex0, baricentric.x * ds);
			triangles.add(r.objectIndex, vertex1, baricentric.y * ds);
			triangles.add(r.objectIndex, vertex2, baricentric.z * ds);
		}

		cells.particles.velocity.set(particleId, velocity);

		// propagate velocities into positions
		cells.particles.position.add(particleId, dt * velocity);

		// zero forces
		cells.particles.force.set(particleId, make_float3(0, 0, 0));
	}

	// 1. Calculate collisions between particles and vein triangles
	// 2. Propagate forces into velocities and velocities into positions. Reset forces to 0 afterwards
	template<>
	__global__ void detectVeinCollisionsAndPropagateParticles<NoGrid>(BloodCells cells, DeviceTriangles triangles, NoGrid  triangleGrid )
	{
		int particleId = blockDim.x * blockIdx.x + threadIdx.x;

		if (particleId >= cells.particleCount)
			return;

		// propagate force into velocities
		float3 F = cells.particles.force.get(particleId);
		float3 velocity = cells.particles.velocity.get(particleId);
		float3 pos = cells.particles.position.get(particleId);

		// upper and lower bound
		if (pos.y >= 0.9f * height)
			velocity.y -= 5;

		if (pos.y <= 0.1f * height)
			velocity.y += 5;

		velocity = velocity + dt * F;
		float3 velocityDir = normalize(velocity);
		ray r(pos, velocityDir);
		float3 reflectedVelociy = make_float3(0, 0, 0);

		bool collicionOccured = false;

		for (int triangleId = 0; triangleId < triangles.triangleCount; ++triangleId)
		{
			// triangle vectices and edges
			float3 v0 = triangles.get(triangleId, vertex0);
			float3 v1 = triangles.get(triangleId, vertex1);
			float3 v2 = triangles.get(triangleId, vertex2);

			if (!(realCollisionDetection(v0, v1, v2, r, reflectedVelociy) 
				&& length(pos - (pos + r.t * r.direction)) <= veinImpactDistance))
				continue;

			r.objectIndex = triangleId;
			collicionOccured = true;
			break;
		}

		if (collicionOccured)
		{
			// triangles move vector, 2 is experimentall constant
			float3 ds = 0.8f * velocityDir;

			float speed = length(velocity);
			velocity = velocityCollisionDamping * speed * reflectedVelociy;

			float3 v0 = triangles.get(r.objectIndex, vertex0);
			float3 v1 = triangles.get(r.objectIndex, vertex1);
			float3 v2 = triangles.get(r.objectIndex, vertex2);
			float3 baricentric = calculateBaricentric(pos + r.t * r.direction, v0, v1, v2);

			// move triangle a bit
			triangles.add(r.objectIndex, vertex0, baricentric.x * ds);
			triangles.add(r.objectIndex, vertex1, baricentric.y * ds);
			triangles.add(r.objectIndex, vertex2, baricentric.z * ds);
		}

		cells.particles.velocity.set(particleId, velocity);

		// propagate velocities into positions
		cells.particles.position.add(particleId, dt * velocity);

		// zero forces
		cells.particles.force.set(particleId, make_float3(0, 0, 0));
	}

	template<int xMin, int xMax, int yMin, int yMax, int zMin, int zMax>
	__device__ bool calculateSideCollisions(float3 position, ray& velocityRay, float3& reflectionVector, DeviceTriangles& triangles, UniformGrid& triangleGrid )
	{
		unsigned int cellId = triangleGrid.calculateCellId(position);

		#pragma unroll
		for (int x = xMin; x <= xMax; x++)
		{
			#pragma unroll	
			for (int y = yMin; y <= yMax; y++)
			{
				#pragma unroll
				for (int z = zMin; z <= zMax; z++)
				{
					int neighborCellId = cellId + z * cellCountX * cellCountY + y * cellCountX + x;

					for (int i = triangleGrid.gridCellStarts[neighborCellId]; i <= triangleGrid.gridCellEnds[neighborCellId]; i++)
					{
						// triangle vectices and edges
						unsigned int triangleId = triangleGrid.particleIds[i];
						float3 v0 = triangles.get(triangleId, vertex0);
						float3 v1 = triangles.get(triangleId, vertex1);
						float3 v2 = triangles.get(triangleId, vertex2);

						if (!(realCollisionDetection(v0, v1, v2, velocityRay, reflectionVector) 
							&& length(position - (position + velocityRay.t * velocityRay.direction)) <= veinImpactDistance))
							continue;

						velocityRay.objectIndex = triangleId;
						return true;
					}
				}
			}
		}
		return false;
	}
}