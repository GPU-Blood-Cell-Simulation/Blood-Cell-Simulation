#pragma once
#include "../defines.hpp"
#include "../utilities/cuda_vec3.cuh"
#include "../utilities/vertex_index_enum.h"
#include "../utilities/math.cuh"
#include "../graphics/mesh.hpp"

#include <vector>
#include <tuple>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/// <summary>
/// Vein triangles
/// </summary>
class DeviceTriangles
{
public:
	const unsigned int triangleCount;
	const unsigned int vertexCount;

	cudaVec3 position;
	cudaVec3 velocity;
	cudaVec3 force;

	unsigned int* indices;
	cudaVec3 centers;

	const float veinVertexHorizontalDistance;
	const float veinVertexNonHorizontalDistances[3];

	DeviceTriangles(const Mesh& mesh, const std::tuple<float, float, float>& springLengths);
	DeviceTriangles(const DeviceTriangles& other);
	~DeviceTriangles();


	__device__ inline unsigned int getIndex(int triangleIndex, VertexIndex vertexIndex) const
	{
		return indices[3 * triangleIndex + vertexIndex];
	}

	__device__ inline float calculateVeinSpringForce(float3 p1, float3 p2, float3 v1, float3 v2, float springLength)
	{
		return (length(p1 - p2) - springLength)* vein_k_sniff + dot(normalize(p1 - p2), (v1 - v2)) * vein_d_fact;
	}

	void gatherForcesFromNeighbors();
	void propagateForcesIntoPositions();

	void calculateCenters();

private:
	bool isCopy = false;

	const unsigned int threadsPerBlock;
	const unsigned int blDim;

	/// !!! STILL NOT IMPLEMENTED !!!
	/// <param name="vertexIndex">0, 1 or 2 as triangle vertices</param>
	/// <returns></returns>
	__device__ inline void atomicAdd(int triangleIndex, VertexIndex vertexIndex, float3 value)
	{
		int index = indices[3 * triangleIndex + vertexIndex];
		position.atomicAddVec3(index, value);
	}
};