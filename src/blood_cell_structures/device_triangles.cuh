#pragma once
#include "../defines.hpp"
#include "../utilities/cuda_vec3.cuh"
#include "../utilities/vertex_index_enum.h"
#include "../graphics/mesh.hpp"

#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/// <summary>
/// Vein triangles
/// </summary>
class DeviceTriangles
{
	bool isCopy = false;
public:
	int triangleCount;
	int vertexCount;

	cudaVec3 vertices;
	int* indices;
	cudaVec3 centers;

	DeviceTriangles(const Mesh& mesh);
	DeviceTriangles(const DeviceTriangles& other);
	~DeviceTriangles();

	/// <param name="vertexIndex">0, 1 or 2 as triangle vertices</param>
	/// <returns></returns>
	__device__ inline float3 get(int triangleIndex, VertexIndex vertexIndex) const
	{
		int index = indices[3 * triangleIndex + vertexIndex];
		return vertices.get(index);
	}

	/// <param name="vertexIndex">0, 1 or 2 as triangle vertices</param>
	/// <returns></returns>
	__device__ inline void set(int triangleIndex, VertexIndex vertexIndex, float3 value)
	{
		int index = indices[3 * triangleIndex + vertexIndex];
		vertices.set(index, value);
	}

	/// <param name="vertexIndex">0, 1 or 2 as triangle vertices</param>
	/// <returns></returns>
	__device__ inline void add(int triangleIndex, VertexIndex vertexIndex, float3 value)
	{
		int index = indices[3 * triangleIndex + vertexIndex];
		vertices.add(index, value);
	}
};