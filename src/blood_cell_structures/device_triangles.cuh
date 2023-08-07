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
	unsigned int triangleCount;
	unsigned int vertexCount;

	cudaVec3 position;
	cudaVec3 velocity;
	cudaVec3 force;

	unsigned int* indices;
	cudaVec3 centers;

	DeviceTriangles(const Mesh& mesh);
	DeviceTriangles(const DeviceTriangles& other);
	~DeviceTriangles();


	__device__ inline unsigned int getIndex(int triangleIndex, VertexIndex vertexIndex) const
	{
		return indices[3 * triangleIndex + vertexIndex];
	}
};