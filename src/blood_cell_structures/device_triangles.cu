#include "device_triangles.cuh"

#include "../utilities/cuda_handle_error.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>


__global__ void calculateCentersKernel(cudaVec3 vertices, int* indices, cudaVec3 centers, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= size)
		return;
	float3 vv1 = vertices.get(indices[3 * id]);
	float3 vv2 = vertices.get(indices[3 * id + 1]);
	float3 vv3 = vertices.get(indices[3 * id + 2]);

	float x = (vv1.x + vv2.x + vv3.x) / 3;
	float y = (vv1.y + vv2.y + vv3.y) / 3;
	float z = (vv1.z + vv2.z + vv3.z) / 3;
	centers.set(id, make_float3(x, y, z));
}

void DeviceTriangles::calculateCenters()
{
	int threadsPerBlock = triangleCount > 1024 ? 1024 : triangleCount;
	int blocks = (triangleCount + threadsPerBlock - 1) / threadsPerBlock;
	calculateCentersKernel << <blocks, threadsPerBlock >> > (vertices, indices, centers, triangleCount);
}

DeviceTriangles::DeviceTriangles(const Mesh& mesh) : triangleCount(mesh.indices.size() / 3), vertexCount(mesh.vertices.size()),
	centers(triangleCount), vertices(vertexCount)
{
	// allocate
	HANDLE_ERROR(cudaMalloc((void**)&indices, 3 * triangleCount * sizeof(int)));

	std::vector<unsigned int> indicesMem = mesh.indices;

	std::vector<float> vx(vertexCount);
	std::vector<float> vy(vertexCount);
	std::vector<float> vz(vertexCount);

	int iter = 0;
	std::for_each(mesh.vertices.begin(), mesh.vertices.end(), [&](auto& v)
		{
		vx[iter] = v.Position.x;
		vy[iter] = v.Position.y;
		vz[iter++] = v.Position.z;
		});

	// copy
	HANDLE_ERROR(cudaMemcpy(indices, indicesMem.data(), 3 * triangleCount * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(vertices.x, vx.data(), vertexCount * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(vertices.y, vy.data(), vertexCount * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(vertices.z, vz.data(), vertexCount * sizeof(float), cudaMemcpyHostToDevice));

	// centers
	calculateCenters();
}

DeviceTriangles::DeviceTriangles(const DeviceTriangles& other) : isCopy(true), triangleCount(other.triangleCount), vertexCount(other.vertexCount),
	vertices(other.vertices), indices(other.indices), centers(other.centers) {}

DeviceTriangles::~DeviceTriangles()
{
	if (!isCopy)
	{
		HANDLE_ERROR(cudaFree(indices));
	}
}
