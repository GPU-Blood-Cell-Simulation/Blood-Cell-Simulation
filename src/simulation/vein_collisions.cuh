#pragma once
#include "../utilities/math.cuh"
#include "../objects/blood_cells.cuh"
#include "../objects/vein_triangles.cuh"
#include "../grids/uniform_grid.cuh"
#include "../grids/octree_grid.cuh"
#include "../grids/no_grid.cuh"
#include "../grids/octree_grid.cuh"
#include <variant>


#include <cmath>

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

using Grid = std::variant<UniformGrid*, NoGrid*, OctreeGrid*>;

namespace sim
{
	// a helper struct for calculating triangle intersections
	struct ray
	{
		float3 origin;
		float3 direction;
		float t = 1e10f;

		// rays may be used to determine intersection with objects
		// so its easy to store object index inside ray
		unsigned int objectIndex = 0;

		__device__ ray(float3 origin, float3 direction);
	};

	__device__ float3 calculateBaricentric(float3 point, float3 v1, float3 v2, float3 v3);

	#pragma region Main Collision Template Kernels
	template<typename T1, typename T2>
	__global__ void detectVeinCollisionsAndPropagateParticles(BloodCells cells, VeinTriangles triangles, T1 particleGrid, T2 triangleGrid) {}

	template<>
	__global__ void detectVeinCollisionsAndPropagateParticles<UniformGrid, NoGrid>(BloodCells cells, VeinTriangles triangles, UniformGrid particleGrid, NoGrid triangleGrid);

	template<>
	__global__ void detectVeinCollisionsAndPropagateParticles<UniformGrid, UniformGrid>(BloodCells cells, VeinTriangles triangles, UniformGrid particleGrid, UniformGrid triangleGrid);

	template<>
	__global__ void detectVeinCollisionsAndPropagateParticles<UniformGrid, OctreeGrid>(BloodCells cells, VeinTriangles triangles, UniformGrid particleGrid, OctreeGrid triangleGrid);
	#pragma endregion

	template<int xMin, int xMax, int yMin, int yMax, int zMin, int zMax>
	__device__ bool calculateSideCollisions(float3 position, ray& r, float3& reflectionVector, VeinTriangles& triangles, UniformGrid& triangleGrid);


	__device__ bool realCollisionDetection(float3 v0, float3 v1, float3 v2, ray& r, float3& reflectionVector);

	__device__ float3 calculateBaricentric(float3 point, float3 v0, float3 v1, float3 v2);

	__device__ bool modifyVelocityIfPositionOutOfBounds(float3& position, float3& velocity, float3 velocityNormalized);
}
