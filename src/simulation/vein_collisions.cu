#include "vein_collisions.cuh"
#include "../utilities/vertex_index_enum.h"
#include "../utilities/cuda_handle_error.cuh"
#include "octree_helper.cuh"

namespace sim
{
	//using octreeHelpers::positive_float_structure;

	__device__ ray::ray(float3 origin, float3 direction) : origin(origin), direction(direction) {}

	__device__ bool realCollisionDetection(float3 v0, float3 v1, float3 v2, ray& r, float3& reflectionVector)
	{
		constexpr float EPS = 0.000001f;
		const float3 edge1 = v1 - v0;
		const float3 edge2 = v2 - v0;

		const float3 h = cross(r.direction, edge2);
		const float a = dot(edge1, h);
		if (a > -EPS && a < EPS)
			return false; // ray parallel to triangle

		const float f = 1 / a;
		const float3 s = r.origin - v0;
		const float u = f * dot(s, h);
		if (u < 0 || u > 1)
			return false;
		const float3 q = cross(s, edge1);
		const float v = f * dot(r.direction, q);
		if (v < 0 || u + v > 1)
			return false;
		const float t = f * dot(edge2, q);
		if (t > EPS)
		{
			r.t = t;

			// this normal is oriented to the vein interior
			// it is caused by the order of vertices in triangles used to correct face culling
			// change order of edge2 and edge1 in cross product for oposite normal
			// Question: Is the situation when we should use oposite normal possible ?
			float3 normal = normalize(cross(edge2, edge1));
			reflectionVector = r.direction - 2 * dot(r.direction, normal) * normal;
			return true;
		}
		return false;
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


	/// <summary>
	/// Function to simulate traversal through octree grid
	/// </summary>
	/// <param name="origin">- ray origin defined in world coords</param>
	/// <param name="direction">- ray direction</param>
	/// <param name="end">- ray end defined in world coords</param>
	/// <param name="masks">- octree nodes masks</param>
	/// <param name="treeData">- pointers to leaves</param>
	/// <param name="maxLevel">- octree depth in levels</param>
	/// <returns>if collision occured with collision data reference parameter</returns>
	__device__ bool traverseGrid(ray r, float3 end, OctreeGrid& grid, VeinTriangles& triangles, float3& reflectionVector)
	{
		// necessary parameters
		const float3 bounding = make_float3(width, height, depth);
		const uint8_t s_max = grid.levels;
		const unsigned int leafShift = (ldexp((double)1, 3 * (grid.levels - 1)) - 1) / 7; // (8^(maxL - 1) - 1)/7

		const float3 normalizedOrigin = r.origin / bounding;
		const float3 normalizedEnd = end / bounding;
		const float3 normalizedDirection = r.direction / bounding;

		const float3 directionSigns = make_float3(!(r.direction.x < 0), !(r.direction.y < 0), !(r.direction.z < 0)); // 1 plus or zero, 0 minus
		const float3 inversedDirection = make_float3(1.0f / normalizedDirection.x, 1.0f / normalizedDirection.y, 1.0f / normalizedDirection.z);

		// values
		unsigned int parentId = 0; // root
		float3 pos = make_float3(0.5f, 0.5f, 0.5f); // initial pos is center of root
		uint8_t scale = s_max - 1;
		uint8_t childId = octreeHelpers::calculateCellForPosition(normalizedOrigin, pos);
		unsigned int realChildId = 8 * parentId + childId + 1;
		float childCellSize = .5f; // normalized cell size

		float3 t = make_float3(0, 0, 0);
		float tEndSquare = length_squared(normalizedEnd - normalizedOrigin);

		while (true) {

			// traversing down the tree to the leaf
			if (scale > 1) {
				childCellSize = 0.5f * childCellSize;
				parentId = realChildId;
				pos = octreeHelpers::calculateChildCenter(pos, childId, childCellSize);
				childId = octreeHelpers::calculateCellForPosition(normalizedOrigin, pos);
				unsigned int realChildId = 8 * parentId + childId + 1;

				if (!(grid.masks[realChildId] & (1 << childId))) // empty cell
					break;

				scale--;
				continue;
			}

			// compute intersections in current cell
			unsigned int leafIndex = realChildId - leafShift;
			unsigned int leafId = grid.treeData[leafIndex];
			unsigned int cellId = grid.gridCellIds[leafId];
			for (int i = leafId; grid.gridCellIds[i] == cellId; ++i)
			{
				// triangle vectices and edges
				unsigned int triangleId = grid.particleIds[i];
				float3 v0 = triangles.positions.get(triangles.getIndex(triangleId, vertex0));
				float3 v1 = triangles.positions.get(triangles.getIndex(triangleId, vertex1));
				float3 v2 = triangles.positions.get(triangles.getIndex(triangleId, vertex2));

				if (!realCollisionDetection(v0, v1, v2, r, reflectionVector))
					continue;

				r.objectIndex = triangleId;
				return true;
			}


			//float3 cellBegining = calculateLeafCellFromMorton(scale, bounding, leafMortonCode, currentLevel);
			float3 cellBeginning = pos - make_float3(fmodf(pos.x, childCellSize),
				fmod(pos.y, childCellSize), fmod(pos.z, childCellSize));

			// maybe conditions instead of directionSigns ???
			t = octreeHelpers::calculateRayTValue(normalizedOrigin, inversedDirection, cellBeginning + childCellSize * directionSigns);
			float tMax = vmin(t);

			// break if ray ends before next cell
			if (tMax > tEndSquare)
				break;

			bool changeParent = false;
			unsigned char bitChange = 0;

			// bit changing && should be + && is minus
			if (!(tMax > t.x) && (childId & 1) && r.direction.x < 0) {
				changeParent = true;
			}
			else if (!(tMax > t.x)) {
				bitChange = 1;
				if ((childId & 1) && r.direction.y < 0) {
					changeParent = true;
				}
			}
			else if (!(tMax > t.x)) {
				bitChange = 2;
				if ((childId & 1) && r.direction.z < 0) {
					changeParent = true;
				}
			}

			if (changeParent) {
				// calculate new pos
				float3 newPos = octreeHelpers::calculateNeighbourLeafPos(pos, normalizedDirection, childCellSize, bitChange);

				octreeHelpers::positive_float_structure posX(pos.x), posY(pos.y), posZ(pos.z);
				octreeHelpers::positive_float_structure newPosX(newPos.x), newPosY(newPos.y), newPosZ(newPos.z);

				uint8_t minMantisaShift = 31 - log2((double)max(posX.mantis ^ newPosX.mantis,
					max(posY.mantis ^ newPosY.mantis, posZ.mantis ^ newPosZ.mantis)));

				scale = s_max - minMantisaShift - 1;
				childCellSize = ldexp((double)1, scale - s_max);

				realChildId = 0;
				unsigned int shift = 0x80000000;
				uint8_t mask = 0;
				for (int i = 31; i > 31 - minMantisaShift; --i) {
					mask = (newPosX.mantis & shift) |
						((newPosY.mantis & shift) << 1) | ((newPosZ.mantis & shift) << 2);
					realChildId += mask;
					realChildId *= 8;
					shift >>= 1;
				}
				childId = mask; // last mask value;
				parentId = (unsigned int)(realChildId * 0.125f); // divide by 8

			}
			else {

				// calculate new childId
				childId ^= 1 << bitChange;
				unsigned int realChildId = 8 * parentId + childId + 1;

				if (!(grid.masks[realChildId] & (1 << childId))) // empty cell
					break;

				// calculate new pos
				pos = octreeHelpers::calculateNeighbourLeafPos(pos, normalizedDirection, childCellSize, bitChange);
			}
		}
		return false;
	}


	template<>
	__global__ void detectVeinCollisionsAndPropagateParticles<OctreeGrid>(BloodCells cells, VeinTriangles triangles,  OctreeGrid triangleGrid)
	{
		int particleId = blockDim.x * blockIdx.x + threadIdx.x;

		if (particleId >= particleCount)
			return;

		float3 F = cells.particles.forces.get(particleId);
		float3 velocity = cells.particles.velocities.get(particleId);
		float3 pos = cells.particles.positions.get(particleId);

		// upper and lower bound
		if (pos.y >= 0.9f * height)
			velocity.y -= 5;

		if (pos.y <= 0.1f * height)
			velocity.y += 5;

		// TEST
		//velocity = velocity + float3{ 0, 0.1f , 0 };
		//return;


		// propagate particle forces into velocities
		velocity = velocity + dt * F;

		// TODO: is there a faster way to calculate this?
		/*if (velocity.x != 0 && velocity.y != 0 && velocity.z != 0)
			goto set_particle_values;*/

		float3 velocityDir = normalize(velocity);

		// cubical bounds
		if (modifyVelocityIfPositionOutOfBounds(pos, velocity, velocityDir)) {
			goto set_particle_values_octree;
		}

		ray r(pos, velocityDir);
		float3 reflectedVelociy = make_float3(0, 0, 0);
		bool collisionOccured = traverseGrid(r, pos + dt*velocity, triangleGrid, triangles, reflectedVelociy);

		if (collisionOccured && length(pos - (pos + r.t * r.direction)) <= 5.0f)
		{
			float3 ds = 0.8f * velocityDir;
			float speed = length(velocity);
			velocity = velocityCollisionDamping * speed * reflectedVelociy;

			unsigned int vertexIndex0 = triangles.getIndex(r.objectIndex, vertex0);
			unsigned int vertexIndex1 = triangles.getIndex(r.objectIndex, vertex1);
			unsigned int vertexIndex2 = triangles.getIndex(r.objectIndex, vertex2);

			float3 v0 = triangles.positions.get(vertexIndex0);
			float3 v1 = triangles.positions.get(vertexIndex1);
			float3 v2 = triangles.positions.get(vertexIndex2);

			float3 baricentric = calculateBaricentric(pos + r.t * r.direction, v0, v1, v2);

			// TODO:
			// Can these lines generate concurrent write conflicts? Unlikely but not impossible. Think about it. - Filip
			// Here we probably should use atomicAdd. - Hubert
			// move triangle a bit
			triangles.forces.add(vertexIndex0, baricentric.x * ds);
			triangles.forces.add(vertexIndex1, baricentric.y * ds);
			triangles.forces.add(vertexIndex2, baricentric.z * ds);

		}

set_particle_values_octree:

		cells.particles.velocities.set(particleId, velocity);

		// propagate velocities into positions
		cells.particles.positions.add(particleId, dt * velocity);

		// zero forces
		cells.particles.forces.set(particleId, make_float3(0, 0, 0));
	}
	
	// 1. Calculate collisions between particles and vein triangles
	// 2. Propagate forces into velocities and velocities into positions. Reset forces to 0 afterwards
	template<>
	__global__ void detectVeinCollisionsAndPropagateParticles<UniformGrid>(BloodCells bloodCells, VeinTriangles triangles, UniformGrid triangleGrid)
	{
		int particleId = blockDim.x * blockIdx.x + threadIdx.x;

		if (particleId >= particleCount)
			return;

		float3 F = bloodCells.particles.forces.get(particleId);
		float3 velocity = bloodCells.particles.velocities.get(particleId);
		float3 pos = bloodCells.particles.positions.get(particleId);

		// TEST
		//velocity = velocity + float3{ 0, 0.1f , 0 };
		//return;
		

		// propagate particle forces into velocities
		velocity = velocity + dt * F;
		
		// TODO: is there a faster way to calculate this?
		/*if (velocity.x != 0 && velocity.y != 0 && velocity.z != 0)
			goto set_particle_values;*/

		float3 velocityDir = normalize(velocity);

		// cubical bounds
		if (modifyVelocityIfPositionOutOfBounds(pos, velocity, velocityDir)) {
			goto set_particle_values_uniform_grid;
		}

		ray r(pos, velocityDir);
		float3 reflectedVelociy = make_float3(0, 0, 0);

		bool collisionDetected = false;
		unsigned int cellId = triangleGrid.calculateCellId(pos);
		unsigned int xId = static_cast<unsigned int>(pos.x / triangleGrid.cellWidth);
		unsigned int yId = static_cast<unsigned int>(pos.y / triangleGrid.cellHeight);
		unsigned int zId = static_cast<unsigned int>(pos.z / triangleGrid.cellDepth);

		// Check all corner cases and call the appropriate function specialization
		// Ugly but fast
		if (xId < 1)
		{
			if (yId < 1)
			{
				if (zId < 1)
				{
					collisionDetected = calculateSideCollisions<0, 1, 0, 1, 0, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else if (zId > triangleGrid.cellCountZ - 2)
				{
					collisionDetected = calculateSideCollisions<0, 1, 0, 1, -1, 0>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else
				{
					collisionDetected = calculateSideCollisions<0, 1, 0, 1, -1, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
			}
			else if (yId > triangleGrid.cellCountY - 2)
			{
				if (zId < 1)
				{
					collisionDetected = calculateSideCollisions<0, 1, -1, 0, 0, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else if (zId > triangleGrid.cellCountZ - 2)
				{
					collisionDetected = calculateSideCollisions<0, 1, -1, 0, -1, 0>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else
				{
					collisionDetected = calculateSideCollisions<0, 1, -1, 0, -1, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
			}
			else
			{
				if (zId < 1)
				{
					collisionDetected = calculateSideCollisions<0, 1, -1, 1, 0, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else if (zId > triangleGrid.cellCountZ - 2)
				{
					collisionDetected = calculateSideCollisions<0, 1, -1, 1, -1, 0>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else
				{
					collisionDetected = calculateSideCollisions<0, 1, -1, 1, -1, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
			}
		}
		else if (xId > triangleGrid.cellCountX - 2)
		{
			if (yId < 1)
			{
				if (zId < 1)
				{
					collisionDetected = calculateSideCollisions<-1, 0, 0, 1, 0, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else if (zId > triangleGrid.cellCountZ - 2)
				{
					collisionDetected = calculateSideCollisions<-1, 0, 0, 1, -1, 0>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else
				{
					collisionDetected = calculateSideCollisions<-1, 0, 0, 1, -1, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
			}
			else if (yId > triangleGrid.cellCountY - 2)
			{
				if (zId < 1)
				{
					collisionDetected = calculateSideCollisions<-1, 0, -1, 0, 0, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else if (zId > triangleGrid.cellCountZ - 2)
				{
					collisionDetected = calculateSideCollisions<-1, 0, -1, 0, -1, 0>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else
				{
					collisionDetected = calculateSideCollisions<-1, 0, -1, 0, -1, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
			}
			else
			{
				if (zId < 1)
				{
					collisionDetected = calculateSideCollisions<-1, 0, -1, 1, 0, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else if (zId > triangleGrid.cellCountZ - 2)
				{
					collisionDetected = calculateSideCollisions<-1, 0, -1, 1, -1, 0>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else
				{
					collisionDetected = calculateSideCollisions<-1, 0, -1, 1, -1, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
			}
		}
		else
		{
			if (yId < 1)
			{
				if (zId < 1)
				{
					collisionDetected = calculateSideCollisions<-1, 1, 0, 1, 0, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else if (zId > triangleGrid.cellCountZ - 2)
				{
					collisionDetected = calculateSideCollisions<-1, 1, 0, 1, -1, 0>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else
				{
					collisionDetected = calculateSideCollisions<-1, 1, 0, 1, -1, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
			}
			else if (yId > triangleGrid.cellCountY - 2)
			{
				if (zId < 1)
				{
					collisionDetected = calculateSideCollisions<-1, 1, -1, 0, 0, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else if (zId > triangleGrid.cellCountZ - 2)
				{
					collisionDetected = calculateSideCollisions<-1, 1, -1, 0, -1, 0>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else
				{
					collisionDetected = calculateSideCollisions<-1, 1, -1, 0, -1, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
			}
			else
			{
				if (zId < 1)
				{
					collisionDetected = calculateSideCollisions<-1, 1, -1, 1, 0, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else if (zId > triangleGrid.cellCountZ - 2)
				{
					collisionDetected = calculateSideCollisions<-1, 1, -1, 1, -1, 0>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
				else
				{
					collisionDetected = calculateSideCollisions<-1, 1, -1, 1, -1, 1>(pos, r, reflectedVelociy, triangles, triangleGrid);
				}
			}
		}

		if (collisionDetected && length(pos - (pos + r.t * r.direction)) <= 5.0f)
		{
			float3 ds = 0.8f * velocityDir;
			float speed = length(velocity);
			velocity = velocityCollisionDamping * speed * reflectedVelociy;

			unsigned int vertexIndex0 = triangles.getIndex(r.objectIndex, vertex0);
			unsigned int vertexIndex1 = triangles.getIndex(r.objectIndex, vertex1);
			unsigned int vertexIndex2 = triangles.getIndex(r.objectIndex, vertex2);

			float3 v0 = triangles.positions.get(vertexIndex0);
			float3 v1 = triangles.positions.get(vertexIndex1);
			float3 v2 = triangles.positions.get(vertexIndex2);

			float3 baricentric = calculateBaricentric(pos + r.t * r.direction, v0, v1, v2);

			// TODO:
			// Can these lines generate concurrent write conflicts? Unlikely but not impossible. Think about it. - Filip
			// Here we probably should use atomicAdd. - Hubert
			// move triangle a bit
			triangles.forces.add(vertexIndex0, baricentric.x * ds);
			triangles.forces.add(vertexIndex1, baricentric.y * ds);
			triangles.forces.add(vertexIndex2, baricentric.z * ds);

		}

	set_particle_values_uniform_grid:

		bloodCells.particles.velocities.set(particleId, velocity);

		// propagate velocities into positions
		bloodCells.particles.positions.add(particleId, dt * velocity);
		
		// zero forces
		bloodCells.particles.forces.set(particleId, make_float3(0, 0, 0));
	}

	// 1. Calculate collisions between particles and vein triangles
	// 2. Propagate forces into velocities and velocities into positions. Reset forces to 0 afterwards
	template<>
	__global__ void detectVeinCollisionsAndPropagateParticles<NoGrid>(BloodCells bloodCells, VeinTriangles triangles, NoGrid triangleGrid)
	{
		int particleId = blockDim.x * blockIdx.x + threadIdx.x;

		if (particleId >= particleCount)
			return;

		// propagate force into velocities
		float3 F = bloodCells.particles.forces.get(particleId);
		float3 velocity = bloodCells.particles.velocities.get(particleId);
		float3 pos = bloodCells.particles.positions.get(particleId);

		velocity = velocity + dt * F;
		float3 velocityDir = normalize(velocity);


		// cubical bounds
		if (modifyVelocityIfPositionOutOfBounds(pos, velocity, velocityDir)) {
			goto set_particle_values_no_grid;
		}

		ray r(pos, velocityDir);
		float3 reflectedVelociy = make_float3(0, 0, 0);

		bool collicionOccured = false;
		for (int triangleId = 0; triangleId < triangles.triangleCount; ++triangleId)
		{
			constexpr float EPS = 1e-7f;
			// triangle vectices and edges
			float3 v0 = triangles.positions.get(triangles.getIndex(triangleId, vertex0));
			float3 v1 = triangles.positions.get(triangles.getIndex(triangleId, vertex1));
			float3 v2 = triangles.positions.get(triangles.getIndex(triangleId, vertex2));
			const float3 edge1 = v1 - v0;
			const float3 edge2 = v2 - v0;

			const float3 h = cross(r.direction, edge2);
			const float a = dot(edge1, h);
			if (a > -EPS && a < EPS)
				continue; // ray parallel to triangle
			
			const float f = 1 / a;
			const float3 s = r.origin - v0;
			const float u = f * dot(s, h);
			if (u < 0 || u > 1)
				continue;
			if (!realCollisionDetection(v0, v1, v2, r, reflectedVelociy))
				continue;

			r.objectIndex = triangleId;
			collicionOccured = true;
			break;
		}

		if (collicionOccured && length(pos - (pos + r.t * r.direction)) <= 5.0f)
		{
			// triangles move vector, 2 is experimentall constant
			float3 ds = 0.8f * velocityDir;

			float speed = length(velocity);
			velocity = velocityCollisionDamping * speed * reflectedVelociy;

			unsigned int vertexIndex0 = triangles.getIndex(r.objectIndex, vertex0);
			unsigned int vertexIndex1 = triangles.getIndex(r.objectIndex, vertex1);
			unsigned int vertexIndex2 = triangles.getIndex(r.objectIndex, vertex2);

			float3 v0 = triangles.positions.get(vertexIndex0);
			float3 v1 = triangles.positions.get(vertexIndex1);
			float3 v2 = triangles.positions.get(vertexIndex2);
			float3 baricentric = calculateBaricentric(pos + r.t * r.direction, v0, v1, v2);

			// move triangle a bit
			// here we probably should use atomicAdd
			triangles.positions.add(vertexIndex0, baricentric.x * ds);
			triangles.positions.add(vertexIndex1, baricentric.y * ds);
			triangles.positions.add(vertexIndex2, baricentric.z * ds);
		}

	set_particle_values_no_grid:

		bloodCells.particles.velocities.set(particleId, velocity);

		// propagate velocities into positions
		bloodCells.particles.positions.add(particleId, dt * velocity);

		// zero forces
		bloodCells.particles.forces.set(particleId, make_float3(0, 0, 0));
	}


	// because floats cannot be template parameters
	// I have put fixed boundary parameters inside this function: 
	// (xMin=0, xMax=width, xMin=0.15*height, yMax=0.85*height, zMin=0, zMax=depth)
	// Keep in mind that yMin, yMax values are strictly bounded due to
	// position of our vein in cubical space (lower and upper vein bounds are at 0.1 and 0.9 of height)
	// (I took 0.05 margin to support situation of intensified falling out of bloodCells at the both ends of vein)
	// these values might have been changed in the future !
	__device__ bool modifyVelocityIfPositionOutOfBounds(float3& position, float3& velocity, float3 normalizedVelocity)
	{
		// experimental value
		// I had one situation of "Position out of bounds" log from calculateCellId function
		// when EPS was 0.001f
		constexpr float EPS = 0.01f;

		float3 newPosition = position + dt * velocity;

		if (newPosition.x < EPS) {

			float dx = EPS - newPosition.x;
			float takeBackLength = dx / normalizedVelocity.x;
			position = position - takeBackLength * normalizedVelocity;
			velocity.x *= -1;
			return true;
		}
		else if (newPosition.x > width - EPS) {

			float dx = newPosition.x - width + EPS;
			float takeBackLength = dx / normalizedVelocity.x;
			position = position - takeBackLength * normalizedVelocity;
			velocity.x *= -1;
			return true;
		}

		if (newPosition.y < 0.15f * height + EPS) {

			float dy = EPS - newPosition.y;
			float takeBackLength = dy / normalizedVelocity.y;
			position = position - takeBackLength * normalizedVelocity;
			velocity.y *= -1;
			return true;
		}
		else if (newPosition.y > 0.85f * height - EPS) {

			float dy = newPosition.y - height + EPS;
			float takeBackLength = dy / normalizedVelocity.y;
			position = position - takeBackLength * normalizedVelocity;
			velocity.y *= -1;
			return true;
		}

		if (newPosition.z < EPS) {

			float dz = EPS - newPosition.z;
			float takeBackLength = dz / normalizedVelocity.z;
			position = position - takeBackLength * normalizedVelocity;
			velocity.z *= -1;
			return true;
		}
		else if (newPosition.z > depth - EPS) {

			float dz = newPosition.z - depth + EPS;
			float takeBackLength = dz / normalizedVelocity.z;
			position = position - takeBackLength * normalizedVelocity;
			velocity.z *= -1;
			return true;
		}
		return false;
	}

	template<int xMin, int xMax, int yMin, int yMax, int zMin, int zMax>
	__device__ bool calculateSideCollisions(float3 position, ray& r, float3& reflectionVector, VeinTriangles& triangles, UniformGrid& triangleGrid)
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
					int neighborCellId = cellId + z * triangleGrid.cellCountX * triangleGrid.cellCountY + y * triangleGrid.cellCountX + x;
					for (int i = triangleGrid.gridCellStarts[neighborCellId]; i <= triangleGrid.gridCellEnds[neighborCellId]; i++)
					{
						// triangle vectices and edges
						unsigned int triangleId = triangleGrid.particleIds[i];
						float3 v0 = triangles.positions.get(triangles.getIndex(triangleId, vertex0));
						float3 v1 = triangles.positions.get(triangles.getIndex(triangleId, vertex1));
						float3 v2 = triangles.positions.get(triangles.getIndex(triangleId, vertex2));

						if (!realCollisionDetection(v0, v1, v2, r, reflectionVector))
							continue;

						r.objectIndex = triangleId;
						return true;
					}
				}
			}
		}
		return false;
	}

}